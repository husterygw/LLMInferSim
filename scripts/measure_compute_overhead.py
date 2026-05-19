"""测量计算 op 的 per-call overhead (eager vs cudagraph delta).

用途: 校准 HardwareConfig.kernel_overhead (Phase 1 / Stage A).

方法:
    跑一个极小 compute op (1×128 GEMM, 数据小到 GPU 几 µs), 分别在 eager 模式 + cudagraph
    模式下计时. delta = eager - cudagraph ≈ Python/ATen/cuLaunch dispatch overhead.

如果 eager - graph > GPU 真实计算时间(典型小 op), 说明 framework overhead 主导,
该把这个 delta 写到 `kernel_overhead[op_category]` 字段。

跑法:
    python scripts/measure_compute_overhead.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from _meas_common import default_out_path, make_record, print_records_table, write_jsonl


SCRIPT = "measure_compute_overhead"


def time_eager(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    return times[len(times)//2]


def time_graph(fn, calls_per_graph=20, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s_stream = torch.cuda.Stream()
    s_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_stream):
        with torch.cuda.graph(g):
            for _ in range(calls_per_graph):
                fn()
    torch.cuda.current_stream().wait_stream(s_stream)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000 / calls_per_graph)
    times.sort()
    return times[len(times)//2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA n/a", file=sys.stderr); return 1
    device = torch.device("cuda")

    records = []

    # 候选 op (model 里常见 compute 类别, 都用极小 shape 让 GPU work 接近 0)
    OPS = {
        "matmul_small":  lambda: torch.matmul(
            torch.ones(1, 64, device=device, dtype=torch.float16),
            torch.ones(64, 64, device=device, dtype=torch.float16),
        ),
        "matmul_med":  lambda: torch.matmul(
            torch.ones(32, 64, device=device, dtype=torch.float16),
            torch.ones(64, 64, device=device, dtype=torch.float16),
        ),
        "elementwise_add": lambda: torch.add(
            torch.ones(128, device=device),
            torch.ones(128, device=device),
        ),
        "norm_like": lambda: torch.nn.functional.layer_norm(
            torch.randn(8, 128, device=device), (128,),
        ),
    }
    for name, fn in OPS.items():
        try:
            t_eager = time_eager(fn, iters=30, warmup=10)
            t_graph = time_graph(fn, calls_per_graph=20, iters=30, warmup=5)
            delta = t_eager - t_graph
            records.append(make_record(
                SCRIPT, op=name,
                eager_us=round(t_eager, 2), graph_us=round(t_graph, 2),
                framework_overhead_us=round(delta, 2),
            ))
        except Exception as e:
            print(f"{name}: {e}", file=sys.stderr)

    out = Path(args.out) if args.out else default_out_path(SCRIPT)
    write_jsonl(out, records)
    print(f"=== {SCRIPT}  output: {out} ===")
    print_records_table(records, ["op", "eager_us", "graph_us", "framework_overhead_us"])
    if records:
        deltas = [r["framework_overhead_us"] for r in records]
        deltas.sort()
        median = deltas[len(deltas)//2]
        print()
        print(f"median framework overhead per compute call: {median:.2f} µs")
        print("写到 hardware.py: kernel_overhead = {'default': "
              f"{median:.0f}e-6}}  (Stage A 校准后决定)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
