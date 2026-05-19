"""Collective (NCCL) runner — 在 torchrun 起的 N rank 里跑 AllReduce / AllToAll.

调用:
    torchrun --nproc-per-node=N -m collector.distributed.run_collective \\
        --shape-profiles qwen3_30b_a3b --topology concentrated [--ops allreduce alltoall]

每 rank 都参与 collective; 只 rank 0 测时 + 写 JSONL / checkpoint.

topology_hint 只是 label, 真正的物理分布由 `CUDA_VISIBLE_DEVICES=...` + 用户 NUMA
拓扑决定. 用户负责选 GPU 列表 (concentrated: 同 NUMA 4 张; balanced: 跨 NUMA).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Optional

from collector import _bootstrap, checkpoint, env_check, writer
from collector import __version__ as _collector_version
from collector.harness import BenchConfig, measure
from collector.paths import DataPaths
from collector.profiles.registry import load_profile
from collector.registry import REGISTRY
from collector.schemas import (
    Case,
    ErrorRecord,
    ExecutionMode,
    Framework,
    Metrics,
    OpKind,
    ProgressEntry,
    RawRecord,
)


_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


def _torch_dtype(dtype_str: str):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]


def _init_nccl(local_rank: int) -> tuple[int, int]:
    """初始化 torch.distributed NCCL, 返 (rank, world_size)."""
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size()


def _bench_collective(
    params: dict,
    rank: int,
    world_size: int,
) -> tuple[float, float, float, bool, Optional[str]]:
    """跑单 collective case, 返 (p10, p50, p90, used_graph, fallback_reason)."""
    import torch
    import torch.distributed as dist

    dtype = _torch_dtype(params["dtype"])
    msg_bytes = int(params["message_size_bytes"])
    elem_bytes = _DTYPE_BYTES[params["dtype"]]
    n_elems = msg_bytes // elem_bytes
    if n_elems <= 0:
        raise ValueError(f"message_size_bytes too small: {msg_bytes}")

    op = params["op_subtype"]
    device = torch.cuda.current_device()

    if op == "allreduce":
        x = torch.randn(n_elems, dtype=dtype, device=device)

        def kernel_func() -> None:
            dist.all_reduce(x)
    elif op == "alltoall":
        # input shape: [W, n_per_rank]; n_elems must divide by W
        if n_elems % world_size != 0:
            raise ValueError(
                f"alltoall n_elems={n_elems} not divisible by world_size={world_size}"
            )
        n_per = n_elems // world_size
        x = torch.randn(n_elems, dtype=dtype, device=device)
        out = torch.empty_like(x)

        def kernel_func() -> None:
            dist.all_to_all_single(out, x)
    else:
        raise NotImplementedError(f"op_subtype {op!r} not supported")

    # barrier so all ranks reach measure together (avoid skew)
    dist.barrier()

    mode = params.get("execution_mode", "cudagraph")
    if mode == "eager":
        cfg = BenchConfig(n_warmups=3, n_iters=10, use_cuda_graph=False)
    elif mode == "cudagraph":
        cfg = BenchConfig(n_warmups=3, n_iters=10, use_cuda_graph=True,
                          allow_graph_fail=False)
    else:
        raise NotImplementedError(f"execution_mode {mode!r} not supported")

    bench = measure(kernel_func, cfg)
    return (
        bench.latency_us_p10, bench.latency_us_p50, bench.latency_us_p90,
        bench.used_cuda_graph, bench.fallback_reason,
    )


def _build_record(
    case: Case,
    p10: float, p50: float, p90: float,
    used_graph: bool,
    fallback_reason: Optional[str],
    *,
    framework_version: str,
    device_name: str,
) -> RawRecord:
    return RawRecord(
        case_id=case.case_id,
        op_kind=OpKind.COLLECTIVE,
        framework=Framework.VLLM,
        framework_version=framework_version,
        device=device_name,
        execution_mode=(
            ExecutionMode.CUDAGRAPH if used_graph else ExecutionMode.EAGER
        ),
        kernel_source="torch_dist_nccl",
        params=dict(case.params),
        metrics=Metrics(
            latency_us_p50=p50, latency_us_p10=p10, latency_us_p90=p90,
            used_cuda_graph=used_graph, n_warmups=3, n_iters=10,
        ),
        metadata={"fallback_reason": fallback_reason},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collector.distributed.run_collective",
        description="NCCL collective collector (torchrun-launched).",
    )
    parser.add_argument("--shape-profiles", nargs="+", required=True,
                        help="profile names, e.g. qwen3_4b qwen3_30b_a3b")
    parser.add_argument("--topology", default="concentrated",
                        choices=["concentrated", "balanced"],
                        help="label only; actual GPU set comes from CUDA_VISIBLE_DEVICES")
    parser.add_argument("--ops", nargs="+",
                        choices=["allreduce", "alltoall"],
                        default=["allreduce", "alltoall"])
    parser.add_argument("--out", default="collector/data")
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank, world_size = _init_nccl(local_rank)

    # env snapshot (rank 0 only writes manifest)
    snap = env_check.collect_env(collector_version=_collector_version)
    hardware = args.hardware or env_check.auto_hardware_id(snap)
    fv = snap.vllm_version

    _bootstrap.register_defaults()
    entry = REGISTRY.require(OpKind.COLLECTIVE, Framework.VLLM)

    profiles = [load_profile(n) for n in args.shape_profiles]
    paths = DataPaths.from_args(args.out, hardware, Framework.VLLM, fv)
    if rank == 0:
        paths.ensure_dirs()
        env_check.write_manifest(snap, paths.manifest_yaml)

    # cases — rank 0 builds, broadcasts case_id list to all ranks via canonical sort
    from collector.scheduler import _resolve_callable
    get_cases = _resolve_callable(entry.get_cases_module)
    all_cases, sources = get_cases(profiles)

    cases = [
        c for c in all_cases
        if c.params.get("num_gpus") == world_size
        and c.params.get("topology_hint") == args.topology
        and c.params.get("op_subtype") in args.ops
    ]
    cases.sort(key=lambda c: c.case_id)   # deterministic order across ranks

    if rank == 0:
        print(f"[rank 0] world_size={world_size} topology={args.topology} "
              f"cases={len(cases)} hardware={hardware}")

    # checkpoint (rank 0 reads, broadcasts done set)
    cp_path = paths.checkpoint_json(entry.op)
    if rank == 0:
        state = checkpoint.load(cp_path, entry.framework, entry.op)
        cases = checkpoint.filter_cases(state, cases, retry_failed=args.retry_failed)
        if args.limit:
            cases = cases[:args.limit]

    # broadcast case_id list from rank 0
    import torch
    import torch.distributed as dist
    if rank == 0:
        case_ids = [c.case_id for c in cases]
    else:
        case_ids = []
    obj_list = [case_ids]
    dist.broadcast_object_list(obj_list, src=0)
    case_ids = obj_list[0]

    if rank != 0:
        cases_by_id = {c.case_id: c for c in all_cases}
        cases = [cases_by_id[i] for i in case_ids]
    if rank == 0:
        print(f"[rank 0] after checkpoint filter: {len(cases)} cases to run")

    # main loop
    op_path = paths.op_jsonl(entry.op)
    err_path = paths.errors_jsonl(entry.op)
    if rank == 0:
        device_name = torch.cuda.get_device_name(local_rank)

    done = 0
    failed = 0
    t0 = time.time()
    for case in cases:
        try:
            p10, p50, p90, used_graph, fr = _bench_collective(
                case.params, rank, world_size,
            )
            if rank == 0:
                rec = _build_record(case, p10, p50, p90, used_graph, fr,
                                    framework_version=fv, device_name=device_name)
                sp = sources.get(case.case_id)
                if sp:
                    rec.metadata.setdefault("source_profiles", list(sp))
                writer.append_record(op_path, rec)
                checkpoint.mark_done(cp_path, state, case.case_id)
                done += 1
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                err = ErrorRecord(
                    case_id=case.case_id, op_kind=OpKind.COLLECTIVE,
                    framework=Framework.VLLM,
                    error_type=type(e).__name__, error_message=str(e),
                    traceback=traceback.format_exc(),
                    metadata={"world_size": world_size},
                )
                sp = sources.get(case.case_id)
                if sp:
                    err.metadata.setdefault("source_profiles", list(sp))
                writer.append_error(err_path, err)
                checkpoint.mark_failed(cp_path, state, case.case_id)
                failed += 1
            # all ranks must keep moving in lockstep; broken barrier means abort
            dist.barrier()

    if rank == 0:
        dur = time.time() - t0
        writer.append_progress(paths.progress_jsonl, ProgressEntry(
            framework=Framework.VLLM, op_kind=OpKind.COLLECTIVE,
            total=len(cases), done=done, failed=failed,
        ))
        print(f"[rank 0] done={done} failed={failed} {dur:.1f}s")

    dist.destroy_process_group()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
