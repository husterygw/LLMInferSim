"""测量 NCCL collective 的 framework_call_overhead (per-collective, eager-cudagraph delta).

用途: 校准 communication.DEFAULT_FRAMEWORK_CALL_OVERHEAD (Phase 2 / Stage B 第 3 步).

跟 measure_collectives.py 的区别:
    measure_collectives.py 是全 size 扫 (含 algorithm_term 占比变化), 跨配置.
    本脚本只测 **极小 size** (latency 主导域), 直接吃 eager - cudagraph delta
    作为 framework_call_overhead per-collective 真值.

跑法:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29611 \\
        scripts/measure_framework_overhead.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import torch
import torch.distributed as dist
from _meas_common import default_out_path, make_record, print_records_table, write_jsonl


SCRIPT = "measure_framework_overhead"


def time_eager(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort(); return ts[len(ts)//2]


def time_graph(fn, calls_per_graph=20, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(calls_per_graph): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / calls_per_graph)
    ts.sort(); return ts[len(ts)//2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--size", type=int, default=1024, help="small bytes for latency-domain")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)

    nelem = max(1, args.size // 4)
    t = torch.zeros(nelem, device=f"cuda:{local_rank}", dtype=torch.float32)

    # 几个 collective 都测 eager - cudagraph delta
    BUILDERS = {
        "allreduce":     lambda: dist.all_reduce(t, op=dist.ReduceOp.SUM),
        "broadcast":     lambda: dist.broadcast(t, src=0),
    }
    if world >= 2:
        # allgather, reducescatter, alltoall (need n-aware tensors)
        out_list = [torch.empty_like(t) for _ in range(world)]
        BUILDERS["allgather"] = lambda: dist.all_gather(out_list, t)
        if t.numel() >= world:
            chunks = list(t.chunk(world))
            out_rs = torch.empty(t.numel() // world, dtype=t.dtype, device=t.device)
            BUILDERS["reducescatter"] = lambda: dist.reduce_scatter(out_rs, chunks)
            out_a2a = torch.empty_like(t)
            BUILDERS["alltoall"] = lambda: dist.all_to_all_single(out_a2a, t)

    records = []
    for name, fn in BUILDERS.items():
        try:
            t_eager = time_eager(fn, iters=30, warmup=10)
            t_graph = time_graph(fn, calls_per_graph=10, iters=20, warmup=3)
            delta = t_eager - t_graph
            if rank == 0:
                records.append(make_record(
                    SCRIPT, collective=name, n=world, size_bytes=args.size,
                    eager_us=round(t_eager, 2), graph_us=round(t_graph, 2),
                    framework_overhead_us=round(delta, 2),
                ))
        except Exception as e:
            if rank == 0: print(f"{name}: {e}", file=sys.stderr)

    if rank == 0:
        out = Path(args.out) if args.out else default_out_path(SCRIPT)
        write_jsonl(out, records)
        print(f"=== {SCRIPT}  output: {out} ===")
        print_records_table(records, ["collective", "n", "size_bytes",
                                       "eager_us", "graph_us", "framework_overhead_us"])
        print()
        print("→ 写入 llm_infer_sim/core/ops/communication.py "
              "DEFAULT_FRAMEWORK_CALL_OVERHEAD 表.")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
