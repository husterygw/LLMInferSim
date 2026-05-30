"""测量 GPU compute peak (BF16/FP8/FP32) vs vendor spec.

用途: HW spec validation (Phase 0 of CALIBRATION_METHODOLOGY).
不写入 profile, 只 sanity check 我们 hardware.py 里填的 peak_flops_* 是否合理.

跑法:
    python scripts/measure_compute_peak.py

输出: configs/calibration/raw/<HW>/<date>/measure_compute_peak.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本能 import scripts._meas_common (不污染 sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import torch
from _meas_common import default_out_path, make_record, print_records_table, write_jsonl


SCRIPT = "measure_compute_peak"


def time_matmul(dtype: torch.dtype, M: int, K: int, N: int, iters: int = 20,
                warmup: int = 10) -> float:
    device = torch.device("cuda")
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    for _ in range(warmup):
        _ = A @ B
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = A @ B
    end.record()
    torch.cuda.synchronize()
    sec = start.elapsed_time(end) / iters * 1e-3
    return sec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="JSONL output path")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    # 关 TF32, 测纯 FP32 cuda core
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    M, K, N = args.m, args.k, args.n
    flops = 2 * M * K * N

    records = []

    # ---- BF16 Tensor Core ----
    try:
        sec = time_matmul(torch.bfloat16, M, K, N)
        records.append(make_record(
            SCRIPT, dtype="bfloat16", m=M, k=K, n=N,
            time_us=sec * 1e6, tflops=flops / sec / 1e12,
        ))
    except Exception as e:
        print(f"BF16 fail: {e}", file=sys.stderr)

    # ---- FP16 Tensor Core ----
    try:
        sec = time_matmul(torch.float16, M, K, N)
        records.append(make_record(
            SCRIPT, dtype="float16", m=M, k=K, n=N,
            time_us=sec * 1e6, tflops=flops / sec / 1e12,
        ))
    except Exception as e:
        print(f"FP16 fail: {e}", file=sys.stderr)

    # ---- FP32 CUDA core (no TF32) ----
    try:
        sec = time_matmul(torch.float32, M, K, N)
        records.append(make_record(
            SCRIPT, dtype="float32", m=M, k=K, n=N,
            time_us=sec * 1e6, tflops=flops / sec / 1e12,
        ))
    except Exception as e:
        print(f"FP32 fail: {e}", file=sys.stderr)

    out = Path(args.out) if args.out else default_out_path(SCRIPT)
    write_jsonl(out, records)
    print(f"=== {SCRIPT}  output: {out} ===")
    print_records_table(records, ["dtype", "m", "k", "n", "time_us", "tflops"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
