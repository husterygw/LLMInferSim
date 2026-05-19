"""测量 GPU HBM 带宽 vs vendor spec.

用途: HW spec validation (Phase 0).
跑大 vector copy / triad, 测 effective GB/s, 跟 hardware.py mem_bandwidth 比.

跑法:
    python scripts/measure_mem_bw.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from _meas_common import default_out_path, make_record, print_records_table, write_jsonl


SCRIPT = "measure_mem_bw"


def time_op(fn, warmup: int = 10, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e-3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sizes-mb", nargs="+", type=int,
                        default=[16, 64, 256, 1024])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr); return 1

    device = torch.device("cuda")
    records = []
    for size_mb in args.sizes_mb:
        N = size_mb * 1024 * 1024 // 4    # fp32 elements
        # ---- copy (read 1 + write 1) ----
        try:
            a = torch.empty(N, device=device, dtype=torch.float32)
            b = torch.empty(N, device=device, dtype=torch.float32)
            sec = time_op(lambda: b.copy_(a))
            bytes_total = N * 4 * 2  # read + write
            records.append(make_record(
                SCRIPT, op="copy", size_mb=size_mb,
                time_us=sec * 1e6, gbps=bytes_total / sec / 1e9,
            ))
            del a, b
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"copy {size_mb}MB fail: {e}", file=sys.stderr)

        # ---- triad (read 2 + write 1, a + α*b) ----
        try:
            a = torch.empty(N, device=device, dtype=torch.float32)
            b = torch.empty(N, device=device, dtype=torch.float32)
            c = torch.empty(N, device=device, dtype=torch.float32)
            sec = time_op(lambda: torch.add(a, b, alpha=1.001, out=c))
            bytes_total = N * 4 * 3   # 2 reads + 1 write
            records.append(make_record(
                SCRIPT, op="triad", size_mb=size_mb,
                time_us=sec * 1e6, gbps=bytes_total / sec / 1e9,
            ))
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"triad {size_mb}MB fail: {e}", file=sys.stderr)

    out = Path(args.out) if args.out else default_out_path(SCRIPT)
    write_jsonl(out, records)
    print(f"=== {SCRIPT}  output: {out} ===")
    print_records_table(records, ["op", "size_mb", "time_us", "gbps"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
