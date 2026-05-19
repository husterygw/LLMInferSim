"""Collector CLI 入口.

跑法:
    python -m collector.cli --help
    python -m collector.cli --list-ops
    python -m collector.cli --frameworks vllm --ops gemm --dry-run
    python -m collector.cli --frameworks vllm --ops gemm --limit 5

设计:
    - main(argv) 返 exit_code (无 sys.exit 在内部, 利于测试)
    - --list-ops / --dry-run 不需要 GPU
    - 真跑 GPU 时, 通过 version_resolver 选 runner, importlib 加载,
      传给 scheduler.run_op
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from collector import __version__ as _collector_version
from collector import _bootstrap, env_check, scheduler, version_resolver
from collector.paths import DataPaths
from collector.profiles import registry as profile_registry
from collector.registry import REGISTRY
from collector.schemas import (
    Case,
    CollectorEntry,
    Framework,
    OpKind,
    RawRecord,
)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collector",
        description="LLMInferSim collector — 真实硬件算子测量数据采集.",
    )

    # 信息类(不需要 GPU)
    p.add_argument("--list-ops", action="store_true",
                   help="列出注册表里所有 (op, framework) entry, 退出.")
    p.add_argument("--show-env", action="store_true",
                   help="打印 GPU / 软件版本 snapshot, 退出.")

    # 选择
    p.add_argument("--frameworks", nargs="+",
                   choices=[f.value for f in Framework], default=["vllm"],
                   help="跑哪些 framework.")
    p.add_argument("--ops", nargs="+",
                   choices=[o.value for o in OpKind], default=None,
                   help="跑哪些 op. 默认全部已注册.")
    p.add_argument("--shape-profiles", nargs="+",
                   default=None,
                   help="shape 来源 profile (e.g. qwen3_4b qwen3_30b_a3b). "
                        "默认全部已知 profile.")

    # 模式 / 限制
    p.add_argument("--dry-run", action="store_true",
                   help="不真跑, 只过滤 + 计数.")
    p.add_argument("--limit", type=int, default=None,
                   help="每 op 最多跑 N case (debug 用).")
    p.add_argument("--retry-failed", action="store_true",
                   help="重跑 checkpoint 标 failed 的 case.")
    p.add_argument("--device", type=int, default=0,
                   help="单进程模式下用哪张 GPU. 默认 0.")

    # 路径
    p.add_argument("--out", default="collector/data",
                   help="输出根目录. 默认 collector/data.")
    p.add_argument("--hardware", default=None,
                   help="HW id, 例 RTX_4090. None 时自动检测.")

    return p


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------

def _cmd_list_ops() -> int:
    """打印 REGISTRY 全部 entries."""
    entries = REGISTRY.all_entries()
    if not entries:
        print("(registry 空 — 暂未实现任何 op runner. 见 collector/_bootstrap.py)")
        return 0
    print(f"{'op':<12} {'framework':<10} {'multi_gpu':<10} {'output':<20} cases_module")
    print("-" * 90)
    for e in entries:
        print(f"{e.op.value:<12} {e.framework.value:<10} "
              f"{str(e.multi_gpu):<10} {e.output_file:<20} {e.get_cases_module}")
    return 0


def _cmd_show_env(collector_version: str) -> int:
    snap = env_check.collect_env(collector_version=collector_version)
    print(f"python_version       {snap.python_version}")
    print(f"torch_version        {snap.torch_version}")
    print(f"cuda_version         {snap.cuda_version}")
    print(f"nccl_version         {snap.nccl_version}")
    print(f"vllm_version         {snap.vllm_version}")
    print(f"sglang_version       {snap.sglang_version or '(not installed)'}")
    print(f"gpu_name             {snap.gpu_name}")
    print(f"gpu_count            {snap.gpu_count}")
    print(f"driver_version       {snap.driver_version}")
    print(f"lock_freq_mhz        {snap.gpu_lock_freq_mhz}")
    print(f"compute_mode         {snap.gpu_exclusive_compute_mode}")
    print(f"hardware_id (auto)   {env_check.auto_hardware_id(snap)}")
    if snap.warnings:
        print("\nWARNINGS:")
        for w in snap.warnings:
            print(f"  - {w}")
    return 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _select_entries(
    args_ops: list[str] | None,
    args_frameworks: list[str],
) -> list[CollectorEntry]:
    """根据 CLI 选项过滤 REGISTRY."""
    frameworks = {Framework(f) for f in args_frameworks}
    ops = {OpKind(o) for o in args_ops} if args_ops else None

    result = []
    for entry in REGISTRY.all_entries():
        if entry.framework not in frameworks:
            continue
        if ops is not None and entry.op not in ops:
            continue
        result.append(entry)
    return result


def _resolve_profiles(names: list[str] | None) -> list:
    """加载 shape profiles. None → 全部已知 profile."""
    if not names:
        names = profile_registry.list_profile_names()
    return [profile_registry.load_profile(n) for n in names]


# run_case_fn 由 scheduler 自己从 entry.run_case_module 解析 (commit #12 起);
# CLI 不再造 placeholder.


def main(argv: list[str] | None = None) -> int:
    """CLI 入口. 返 exit code (无 sys.exit, 利于测试)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # bootstrap registry
    _bootstrap.register_defaults()

    # 信息子命令
    if args.list_ops:
        return _cmd_list_ops()
    if args.show_env:
        return _cmd_show_env(_collector_version)

    # 主流程: 选 entries
    entries = _select_entries(args.ops, args.frameworks)
    if not entries:
        print(f"No entries matched ops={args.ops}, frameworks={args.frameworks}. "
              f"Try --list-ops to see what's registered.", file=sys.stderr)
        return 2

    # env snapshot
    snap = env_check.collect_env(collector_version=_collector_version)
    hardware = args.hardware or env_check.auto_hardware_id(snap)
    if hardware == "unknown":
        print("WARNING: hardware unknown. Pass --hardware <id> explicitly.",
              file=sys.stderr)

    # 一个 framework 一份 paths (一次跑一个 framework, 多 framework 串行)
    # shape-profile filter: 加载选定 profile, 用于 case 生成 (传给 entry.get_cases_module)
    # NOTE: 当前 case 列表由 entry.get_cases_module 直接返, 还没接 profile 注入.
    # commit 后续完善后, 这里把 profiles 列表传给 get_cases.
    try:
        selected_profiles = _resolve_profiles(args.shape_profiles)
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    overall_done = 0
    overall_failed = 0
    overall_skipped = 0

    for framework_str in args.frameworks:
        framework = Framework(framework_str)
        framework_version = _get_framework_version(framework, snap)

        paths = DataPaths.from_args(
            args.out, hardware, framework, framework_version,
        )

        # 落 manifest
        if not args.dry_run:
            paths.ensure_dirs()
            try:
                env_check.write_manifest(snap, paths.manifest_yaml)
            except ImportError:
                pass    # yaml 缺装的话, 走 silent (不致命)

        # 过滤本 framework 的 entries
        fw_entries = [e for e in entries if e.framework == framework]
        if not fw_entries:
            continue

        # run_case_fn=None → scheduler 从 entry.run_case_module 解析 (version-aware)
        results = scheduler.run_all(
            fw_entries, paths, framework_version,
            profiles=selected_profiles,
            device=args.device,
            retry_failed=args.retry_failed,
            limit=args.limit,
            dry_run=args.dry_run,
            case_filter=None,
        )

        # 打印 summary
        print(f"\n=== framework={framework.value} {'(DRY RUN)' if args.dry_run else ''} ===")
        print(f"{'op':<12} {'total':<7} {'done':<7} {'failed':<7} {'skipped_mp':<11} {'sec':<7}")
        print("-" * 60)
        for (op, _), s in sorted(results.items(), key=lambda x: x[0][0].value):
            mark = "skip" if s.skipped_multi_gpu else "-"
            print(f"{op.value:<12} {s.total_cases:<7} {s.done:<7} "
                  f"{s.failed:<7} {mark:<11} {s.duration_sec:<7.1f}")
            overall_done += s.done
            overall_failed += s.failed
            if s.skipped_multi_gpu:
                overall_skipped += 1

    print(f"\nOVERALL: done={overall_done} failed={overall_failed} "
          f"skipped_multi_gpu={overall_skipped}")
    return 0 if overall_failed == 0 else 1


def _get_framework_version(framework: Framework, snap) -> str:
    """从 env snapshot 取 framework version."""
    if framework == Framework.VLLM:
        return snap.vllm_version
    if framework == Framework.SGLANG:
        return snap.sglang_version
    return ""


if __name__ == "__main__":
    sys.exit(main())
