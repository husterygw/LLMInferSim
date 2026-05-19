"""CLI: python -m llm_infer_sim.calibration ... (详设 §9.4.2 B.3).

子命令:
  profile    — 跑 vLLM layerwise_profile, 输出三类 CSV (dense / attention / per_sequence)
  fit        — CSV → EfficiencyProfile YAML (B.5 加, 当前 stub)

例:
  # 在 RTX 4090 上校 Qwen3-4B bf16
  python -m llm_infer_sim.calibration profile \\
      --model Qwen/Qwen3-4B \\
      --model-type qwen3 \\
      --hardware RTX_4090 \\
      --dtype bfloat16 \\
      --output configs/efficiency/raw

  # 只跑 dense + per_sequence (跳过耗时的 attention 网格)
  python -m llm_infer_sim.calibration profile \\
      --model Qwen/Qwen3-4B --model-type qwen3 --hardware RTX_4090 \\
      --kinds dense per_sequence
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_infer_sim.calibration",
        description="Layer 1 op-kernel microbench calibration (详设 §9.4.2 Plan B)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- profile ----
    p_profile = sub.add_parser(
        "profile", help="跑 vLLM layerwise_profile, 输出三类 CSV"
    )
    p_profile.add_argument(
        "--model", required=True,
        help="HF id 或本地路径 (例 Qwen/Qwen3-4B / /data/models/Qwen3-4B)",
    )
    p_profile.add_argument(
        "--model-type", required=True,
        help="HF config model_type, 用来找 catalog YAML (例 qwen3 / llama)",
    )
    p_profile.add_argument(
        "--hardware", required=True,
        help="自由命名硬件标识, 决定输出目录 (例 RTX_4090 / H100_SXM)",
    )
    p_profile.add_argument(
        "--dtype", default="bfloat16",
        choices=("bfloat16", "float16", "bf16", "fp16", "float32", "auto"),
    )
    p_profile.add_argument(
        "--tp", type=int, default=1, help="tensor_parallel_size (默认 1)",
    )
    p_profile.add_argument(
        "--iterations", type=int, default=3,
        help="每 shot 内 forward 次数 (默认 3, 跨 invocation 取均值)",
    )
    p_profile.add_argument(
        "--output", default="configs/efficiency/raw",
        help="输出 root (默认 configs/efficiency/raw)",
    )
    p_profile.add_argument(
        "--kinds", nargs="+", default=None,
        choices=("dense", "attention", "per_sequence"),
        help="只跑这些 category (默认全跑)",
    )
    p_profile.add_argument(
        "--max-model-len", type=int, default=20480,
        help="vLLM max sequence length (default 20480 容纳 ATTENTION_SHOTS 16k kv 极限)",
    )
    p_profile.add_argument(
        "--max-num-seqs", type=int, default=16,
        help="vLLM max concurrent seqs (default 16 跟 ATTENTION_SHOTS 最大 n_decode 对齐)",
    )
    p_profile.add_argument(
        "--no-resume", action="store_true",
        help="禁用 resume, 不跳过已 visited shot (慢但保证 fresh 数据)",
    )
    p_profile.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )

    # ---- fit ----
    p_fit = sub.add_parser(
        "fit",
        help="CSV → EfficiencyProfile YAML (B.5)",
    )
    p_fit.add_argument("--raw", required=True, help="profile raw 目录")
    p_fit.add_argument("--out", required=True, help="输出 YAML 路径")

    # ---- preflight ----
    p_pre = sub.add_parser(
        "preflight",
        help="跑长 profile 前的快速健康检查 (~1-2 min)",
    )
    p_pre.add_argument("--model", default="Qwen/Qwen3-4B")
    p_pre.add_argument("--model-type", default="qwen3")
    p_pre.add_argument("--hardware", default="RTX_4090")
    p_pre.add_argument("--dtype", default="bfloat16")
    p_pre.add_argument("--test-tokens", type=int, default=8)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level if hasattr(args, "log_level") else "INFO"),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "profile":
        return _cmd_profile(args)
    if args.cmd == "fit":
        return _cmd_fit(args)
    if args.cmd == "preflight":
        return _cmd_preflight(args)
    parser.error(f"unknown cmd: {args.cmd}")


def _cmd_preflight(args) -> int:
    from llm_infer_sim.calibration.preflight import preflight
    return preflight(
        model=args.model, model_type=args.model_type,
        hardware=args.hardware, dtype=args.dtype,
        test_tokens=args.test_tokens,
    )


def _cmd_profile(args) -> int:
    # 防御: VLLM_VIRTUAL_BACKEND=1 会让 vLLM 走 sim 路径, 测不到真 GPU
    import os
    if os.environ.get("VLLM_VIRTUAL_BACKEND") == "1":
        print(
            "[error] VLLM_VIRTUAL_BACKEND=1 设了, calibration 必须真 GPU 路径. "
            "请 `unset VLLM_VIRTUAL_BACKEND` 后再跑.",
            file=sys.stderr,
        )
        return 2

    from llm_infer_sim.calibration.runner import run_calibration

    try:
        out_dir = run_calibration(
            model=args.model,
            model_type=args.model_type,
            hardware=args.hardware,
            dtype=args.dtype,
            output_root=args.output,
            tp=args.tp,
            iterations=args.iterations,
            kinds=tuple(args.kinds) if args.kinds else None,
            resume=not args.no_resume,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
        )
    except KeyboardInterrupt:
        print("[interrupt] 用户中断", file=sys.stderr)
        return 130
    print(f"PASSED. Output: {out_dir}")
    return 0


def _cmd_fit(args) -> int:
    """CSV → EfficiencyProfile YAML (B.5)."""
    from llm_infer_sim.calibration.fit import fit_efficiency
    try:
        profile = fit_efficiency(raw_dir=args.raw, out_yaml=args.out)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    print(
        f"PASSED. Fitted {len(profile.entries)} entries to {args.out}",
        f"(hardware={profile.hardware}, default_compute={profile.default_compute:.3f})",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
