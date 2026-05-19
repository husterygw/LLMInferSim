"""调度核心 — 串行编排 + resume + 错误隔离.

设计:
  - 第一版只串行(num_workers=1), 多进程并发 + restart-on-crash 留下 commit
  - 真 GPU 执行(run_case)通过 callable 注入, 测试时 mock
  - 一个 op 一个 op 跑(避免跨 op cudagraph 池干扰)
  - 一次 case 完成立刻 checkpoint 持久化, crash 也只丢一条
  - dry_run 模式: 只过滤 + 计数, 不真跑

接口:
  run_op(entry, paths, version, run_case_fn) -> RunSummary
  run_all(entries, paths, version, run_case_fn) -> dict[(op, fw)] -> RunSummary
"""
from __future__ import annotations

import importlib
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

from collector import checkpoint, version_resolver, writer
from collector.paths import DataPaths
from collector.profiles._dims import ProfileSpec
from collector.schemas import (
    Case,
    CollectorEntry,
    ErrorRecord,
    Framework,
    OpKind,
    ProgressEntry,
    RawRecord,
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    """单 op 跑完后的统计."""
    op: OpKind
    framework: Framework
    total_cases: int = 0       # checkpoint filter 后剩多少待跑
    done: int = 0               # 本轮新增成功
    failed: int = 0             # 本轮新增失败
    skipped_multi_gpu: bool = False  # entry.multi_gpu=True 时主 scheduler 跳过
    duration_sec: float = 0.0


# ---------------------------------------------------------------------------
# Importlib resolution (entry.get_cases_module / run_case_module → callable)
# ---------------------------------------------------------------------------

def _resolve_callable(spec: str) -> Callable:
    """'pkg.mod:func' → callable. 'pkg.mod' (无冒号) → module 自身的 __call__."""
    if ":" in spec:
        mod_path, func_name = spec.split(":", 1)
    else:
        mod_path, func_name = spec, None
    module = importlib.import_module(mod_path)
    if func_name is None:
        return module    # type: ignore[return-value]
    return getattr(module, func_name)


# ---------------------------------------------------------------------------
# 一个 case 跑一次(成功/失败统一接口)
# ---------------------------------------------------------------------------

def _run_one_case(
    case: Case,
    device: int,
    framework: Framework,
    run_case_fn: Callable[[Case, int], RawRecord],
) -> tuple[Optional[RawRecord], Optional[ErrorRecord]]:
    """跑单 case, 异常包成 ErrorRecord.

    Returns:
        (record, None) 成功; (None, error) 失败. 永远不 raise.
    """
    try:
        record = run_case_fn(case, device)
        return record, None
    except Exception as e:  # noqa: BLE001
        error = ErrorRecord(
            case_id=case.case_id,
            op_kind=case.op_kind,
            framework=framework,
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=traceback.format_exc(),
            metadata={"device": device},
        )
        return None, error


# ---------------------------------------------------------------------------
# 单 op 调度
# ---------------------------------------------------------------------------

def run_op(
    entry: CollectorEntry,
    paths: DataPaths,
    framework_version: str,
    run_case_fn: Optional[Callable[[Case, int], RawRecord]] = None,
    *,
    profiles: Optional[list[ProfileSpec]] = None,
    cases: Optional[list[Case]] = None,
    case_sources: Optional[dict[str, list[str]]] = None,
    device: int = 0,
    retry_failed: bool = False,
    limit: Optional[int] = None,
    dry_run: bool = False,
    case_filter: Optional[Callable[[Case], bool]] = None,
) -> RunSummary:
    """跑一个 (op, framework) entry 全部 case.

    Cases 来源 (优先级):
        1. 显式 `cases=` 入参 (测试 / 外部 case 提供路径)
        2. `entry.get_cases_module(profiles)` (生产路径, 需 profiles 列表)

    Args:
        profiles: shape profile 列表. 必须提供, 除非显式给 `cases`.
        cases: 直接给 case 列表 (跳过 entry.get_cases_module). 测试用.
        case_sources: {case_id: [profile_name, ...]} 来源映射,
                      runner 完成后 scheduler 把它写到 record.metadata["source_profiles"].

    Returns:
        RunSummary
    """
    summary = RunSummary(op=entry.op, framework=entry.framework)
    t0 = time.time()

    # multi_gpu 类(collective)跳过, 由 distributed/ 走 torchrun
    if entry.multi_gpu:
        summary.skipped_multi_gpu = True
        return summary

    # 0. 解析 run_case_fn — 测试可注入 (传 run_case_fn=mock), 否则从 entry 解析
    if run_case_fn is None:
        runner_spec = version_resolver.resolve_runner(entry, framework_version)
        if ":" not in runner_spec:
            runner_spec = f"{runner_spec}:run_case"
        run_case_fn = _resolve_callable(runner_spec)

    # 1. 加载 case 列表 + sources
    if cases is None:
        if profiles is None:
            raise ValueError("Must provide either `cases` or `profiles`")
        get_cases = _resolve_callable(entry.get_cases_module)
        result = get_cases(profiles)
        # 支持 (cases, sources) tuple 或 单纯 list
        if isinstance(result, tuple) and len(result) == 2:
            cases, case_sources = result
        else:
            cases, case_sources = list(result), {}
    if case_sources is None:
        case_sources = {}

    # 2. 外部 case 过滤(留作 --filter 钩子)
    if case_filter is not None:
        cases = [c for c in cases if case_filter(c)]

    # 3. checkpoint 过滤
    cp_path = paths.checkpoint_json(entry.op)
    state = checkpoint.load(cp_path, entry.framework, entry.op)
    cases_to_run = checkpoint.filter_cases(state, cases, retry_failed=retry_failed)

    # 4. limit
    if limit is not None:
        cases_to_run = cases_to_run[:limit]
    summary.total_cases = len(cases_to_run)

    if dry_run:
        summary.duration_sec = time.time() - t0
        return summary

    # 5. 准备落盘
    paths.ensure_dirs()
    op_path = paths.op_jsonl(entry.op)
    err_path = paths.errors_jsonl(entry.op)

    # 6. 逐 case 跑
    for case in cases_to_run:
        record, error = _run_one_case(case, device, entry.framework, run_case_fn)
        if record is not None:
            # 注入 source_profiles provenance 到 metadata
            sp = case_sources.get(case.case_id)
            if sp:
                record.metadata.setdefault("source_profiles", list(sp))
            writer.append_record(op_path, record)
            checkpoint.mark_done(cp_path, state, case.case_id)
            summary.done += 1
        else:
            assert error is not None    # 分支必有其一
            # 错误 record 也带 source_profiles
            sp = case_sources.get(case.case_id)
            if sp:
                error.metadata.setdefault("source_profiles", list(sp))
            writer.append_error(err_path, error)
            checkpoint.mark_failed(cp_path, state, case.case_id)
            summary.failed += 1

    # 7. 写进度
    writer.append_progress(paths.progress_jsonl, ProgressEntry(
        framework=entry.framework,
        op_kind=entry.op,
        total=summary.total_cases,
        done=summary.done,
        failed=summary.failed,
    ))

    summary.duration_sec = time.time() - t0
    return summary


# ---------------------------------------------------------------------------
# 全 entries 调度
# ---------------------------------------------------------------------------

def run_all(
    entries: list[CollectorEntry],
    paths: DataPaths,
    framework_version: str,
    run_case_fn: Optional[Callable[[Case, int], RawRecord]] = None,
    *,
    profiles: Optional[list[ProfileSpec]] = None,
    device: int = 0,
    retry_failed: bool = False,
    limit: Optional[int] = None,
    dry_run: bool = False,
    case_filter: Optional[Callable[[Case], bool]] = None,
) -> dict[tuple[OpKind, Framework], RunSummary]:
    """串行跑全部 entries. 每 op 独立 checkpoint.

    `run_case_fn=None` 时, 各 entry 自行从 entry.run_case_module 解析 runner.
    """
    results: dict[tuple[OpKind, Framework], RunSummary] = {}
    for entry in entries:
        results[(entry.op, entry.framework)] = run_op(
            entry, paths, framework_version, run_case_fn,
            profiles=profiles,
            device=device, retry_failed=retry_failed,
            limit=limit, dry_run=dry_run, case_filter=case_filter,
        )
    return results
