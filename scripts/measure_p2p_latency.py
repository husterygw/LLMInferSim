"""测量 NCCL P2P send/recv 真实 link latency (1-byte cudagraph).

用途: 校准 HardwareConfig.comm_step_latency (Phase 2 / Stage B 依赖链第 1 步).

方法:
    GPU 0 → GPU 1 互相 send 1 byte, cudagraph 模式 (扣掉 framework dispatch).
    `T_total / 2 hops` ≈ effective per-hop latency = comm_step_latency.

跑法:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29610 \\
        scripts/measure_p2p_latency.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.distributed as dist
from _meas_common import default_out_path, make_record, print_records_table, write_jsonl


SCRIPT = "measure_p2p_latency"


def time_graph(fn, calls_per_graph=20, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(calls_per_graph):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000 / calls_per_graph)
    times.sort()
    return times[len(times)//2]


def time_eager(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000)
    times.sort()
    return times[len(times)//2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        if rank == 0: print("requires world_size=2", file=sys.stderr)
        return 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)

    t = torch.zeros(1, device=f"cuda:{local_rank}", dtype=torch.float32)

    if rank == 0:
        def fn():
            dist.send(t, dst=1)
            dist.recv(t, src=1)
    else:
        def fn():
            dist.recv(t, src=0)
            dist.send(t, dst=0)

    # 注: P2P cudagraph 在某些 torch 版本不稳, fallback to eager 也 OK 拿 latency 上限.
    # 这里用 eager (P2P 调用本身 framework overhead 较小, 比 collective 简单).
    rt_eager = time_eager(fn, iters=30, warmup=10)
    # cudagraph 的 P2P 可能失败 → catch
    try:
        rt_graph = time_graph(fn, calls_per_graph=10, iters=20, warmup=3)
    except Exception as e:
        if rank == 0:
            print(f"cudagraph P2P failed ({e}), fall back to eager only", file=sys.stderr)
        rt_graph = None

    if rank == 0:
        records = []
        # eager: 1 round-trip ≈ 2 hops + 2 framework
        records.append(make_record(
            SCRIPT, mode="eager", payload_bytes=4, round_trip_us=round(rt_eager, 2),
            per_hop_us=round(rt_eager / 2, 2),
        ))
        if rt_graph is not None:
            records.append(make_record(
                SCRIPT, mode="cudagraph", payload_bytes=4, round_trip_us=round(rt_graph, 2),
                per_hop_us=round(rt_graph / 2, 2),
            ))
        out = Path(args.out) if args.out else default_out_path(SCRIPT)
        write_jsonl(out, records)
        print(f"=== {SCRIPT}  output: {out} ===")
        print_records_table(records, ["mode", "payload_bytes", "round_trip_us", "per_hop_us"])
        if rt_graph is not None:
            print()
            print(f"comm_step_latency (cudagraph per-hop): {rt_graph/2:.2f} µs")
            print(f"  → 写入 hardware.py 该 HW profile comm_step_latency 字段 (秒): "
                  f"{rt_graph/2 * 1e-6:.1e}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
