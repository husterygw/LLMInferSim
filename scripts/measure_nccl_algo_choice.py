"""测量 NCCL 实际选择的 algorithm (auto vs forced ring/tree).

用途: 验证我们 cost model 用的 min(ring,tree) 是否跟 NCCL 实际行为一致.
不直接生 hardware profile 值; 输出供后续校准 `collective_algo_bias` 时参考.

方法:
    跑同一个 AllReduce/Broadcast, 分别在 NCCL_ALGO=Ring / Tree / 默认 下计时.
    看默认跟哪个最接近, 反推 NCCL 在当前 size 上选了什么.

跑法:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29612 \\
        scripts/measure_nccl_algo_choice.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.distributed as dist
from _meas_common import default_out_path, make_record, print_records_table, write_jsonl


SCRIPT = "measure_nccl_algo_choice"


def time_eager(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort(); return ts[len(ts)//2]


def measure_single(args, size_bytes: int) -> dict:
    """跑当前进程的 measurement (single algo for current NCCL_ALGO env).

    Returns dict {coll, n, size, ...}.
    """
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    nelem = max(1, size_bytes // 4)
    t = torch.zeros(nelem, device=f"cuda:{local_rank}", dtype=torch.float32)
    res = {}
    for coll in ("allreduce", "broadcast"):
        if coll == "allreduce":
            fn = lambda: dist.all_reduce(t, op=dist.ReduceOp.SUM)
        else:
            fn = lambda: dist.broadcast(t, src=0)
        med = time_eager(fn)
        if rank == 0:
            res[coll] = med
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[1024, 65536, 1048576, 16777216])
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    nccl_algo = os.environ.get("NCCL_ALGO", "(auto)")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)

    records = []
    for size in args.sizes:
        meas = measure_single(args, size)
        if rank == 0:
            for coll, med in meas.items():
                records.append(make_record(
                    SCRIPT, collective=coll, n=world, size_bytes=size,
                    nccl_algo_env=nccl_algo, median_us=round(med, 2),
                ))

    if rank == 0:
        out = Path(args.out) if args.out else default_out_path(SCRIPT)
        write_jsonl(out, records)
        print(f"=== {SCRIPT}  NCCL_ALGO={nccl_algo}  output: {out} ===")
        print_records_table(records,
            ["collective", "n", "size_bytes", "nccl_algo_env", "median_us"])
        print()
        print("→ 后续: 跑 NCCL_ALGO=Ring / Tree / 默认 三次, 比 median_us, "
              "看默认选了哪个. 如发现默认在某 size 域不选 min(候选), 反映到 "
              "hw.collective_algo_bias 字段.")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
