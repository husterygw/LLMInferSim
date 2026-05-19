"""Comprehensive NCCL collective communication benchmark.

测 6 个 collective × 多 size × eager/cudagraph 模式, 输出 JSONL 供 analyze_collectives.py
分析。每个 process 独立由 torchrun 拉起, 由 driver script 提供不同的 CUDA_VISIBLE_DEVICES
来覆盖 (n, NUMA) 配置。

跑法 (driver: scripts/run_collective_sweep.sh):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
        scripts/measure_collectives.py \\
        --out /tmp/collective_bench.jsonl --label same_numa_n2

输出每行 JSON:
    {"collective": "allreduce", "n": 2, "size_bytes": 1024, "mode": "eager",
     "median_us": 84.5, "p10_us": 80.1, "p90_us": 92.3, "iters": 30,
     "label": "same_numa_n2"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable

import torch
import torch.distributed as dist


COLLECTIVES = ["allreduce", "allgather", "reducescatter", "alltoall", "broadcast", "p2p"]
DEFAULT_SIZES = [
    16, 64, 256, 1024, 4096, 16384, 65536, 262144,
    1048576, 4194304, 16777216, 67108864, 268435456,
]
DEFAULT_MODES = ["eager", "cudagraph"]


def _percentiles(times_us: list[float]) -> tuple[float, float, float]:
    times_us = sorted(times_us)
    n = len(times_us)
    return times_us[n // 2], times_us[n // 10], times_us[9 * n // 10]


def time_eager(fn: Callable[[], None], iters: int, warmup: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_us = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000)
    return _percentiles(times_us)


def time_cudagraph(
    fn: Callable[[], None], calls_per_graph: int, iters: int, warmup: int
) -> tuple[float, float, float]:
    """Capture fn into CUDA graph (calls_per_graph copies), replay iters times.
    Returns per-call time (total / calls_per_graph)."""
    # Warmup outside graph
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    # Stream context for NCCL graph capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(calls_per_graph):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    times_us = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000 / calls_per_graph)
    return _percentiles(times_us)


def make_fn(coll: str, t: torch.Tensor, n: int, rank: int, local_rank: int) -> Callable[[], None]:
    """Build a callable that runs one invocation of the collective."""
    dev = t.device

    if coll == "allreduce":
        return lambda: dist.all_reduce(t, op=dist.ReduceOp.SUM)

    if coll == "allgather":
        # 输入每 rank t, 输出 list of n tensors
        out_list = [torch.empty_like(t) for _ in range(n)]
        return lambda: dist.all_gather(out_list, t)

    if coll == "reducescatter":
        # input: list of n chunks (size t.numel/n each), output: 1 chunk
        # 简化: 把 t 切 n 份当 input list, output 是 1 份
        per_rank_elem = t.numel() // n
        if per_rank_elem == 0:
            # t 太小, 跳过
            return lambda: None
        out = torch.empty(per_rank_elem, dtype=t.dtype, device=dev)
        chunks = list(t.chunk(n))
        return lambda: dist.reduce_scatter(out, chunks)

    if coll == "alltoall":
        # uniform alltoall: 每 rank send t[i*per:i*per+per] to rank i, recv into out
        per_rank_elem = t.numel() // n
        if per_rank_elem == 0:
            return lambda: None
        out = torch.empty_like(t)
        return lambda: dist.all_to_all_single(out, t)

    if coll == "broadcast":
        return lambda: dist.broadcast(t, src=0)

    if coll == "p2p":
        # 只支持 n=2; round-trip (0→1→0)
        if n != 2:
            return lambda: None
        if rank == 0:
            def fn():
                dist.send(t, dst=1)
                dist.recv(t, src=1)
            return fn
        else:  # rank == 1
            def fn():
                dist.recv(t, src=0)
                dist.send(t, dst=0)
            return fn

    raise ValueError(f"Unknown collective: {coll}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="JSONL output path (append)")
    parser.add_argument("--label", required=True, help="e.g. same_numa_n2 / cross_numa_n4 / full_n8")
    parser.add_argument("--collectives", nargs="+", default=COLLECTIVES,
                        choices=COLLECTIVES + ["all"])
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                        choices=DEFAULT_MODES + ["all"])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--graph-calls", type=int, default=10,
                        help="number of collective calls captured in one CUDA graph")
    args = parser.parse_args()

    if "all" in args.collectives:
        args.collectives = COLLECTIVES
    if "all" in args.modes:
        args.modes = DEFAULT_MODES

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    is_master = (rank == 0)
    out_f = None
    if is_master:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        out_f = open(args.out, "a")
        print(f"[label={args.label}] rank=0/{world_size}, writing → {args.out}", file=sys.stderr)

    for coll in args.collectives:
        if coll == "p2p" and world_size != 2:
            if is_master:
                print(f"  skip {coll} (n={world_size}, p2p only for n=2)", file=sys.stderr)
            continue

        for size in args.sizes:
            nelem = max(1, size // 4)   # fp32 elements
            for mode in args.modes:
                # alloc fresh tensor each (avoid graph state leaks)
                t = torch.ones(nelem, device=f"cuda:{local_rank}", dtype=torch.float32)
                fn = make_fn(coll, t, world_size, rank, local_rank)

                # skip if fn is no-op (size too small for collective like reducescatter)
                # quick check: run once and see (or check upfront)
                if coll in ("reducescatter", "alltoall") and t.numel() < world_size:
                    continue

                try:
                    if mode == "eager":
                        med, p10, p90 = time_eager(fn, args.iters, args.warmup)
                    else:  # cudagraph
                        # p2p in cudagraph: send/recv graph capture 在某些 torch 版本不稳, skip
                        if coll == "p2p":
                            continue
                        med, p10, p90 = time_cudagraph(
                            fn, args.graph_calls, args.iters, max(2, args.warmup // 2)
                        )
                except Exception as e:
                    if is_master:
                        print(f"  ERROR {coll}/{size}/{mode}: {type(e).__name__}: {e}",
                              file=sys.stderr)
                    continue

                if is_master:
                    rec = {
                        "collective": coll,
                        "n": world_size,
                        "size_bytes": int(size),
                        "mode": mode,
                        "median_us": round(med, 3),
                        "p10_us": round(p10, 3),
                        "p90_us": round(p90, 3),
                        "iters": args.iters,
                        "label": args.label,
                    }
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    print(f"  {coll:<15} n={world_size} size={size:>10}B "
                          f"{mode:<10} median={med:>8.1f} us",
                          file=sys.stderr)

    if is_master and out_f is not None:
        out_f.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
