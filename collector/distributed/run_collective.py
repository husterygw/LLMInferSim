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


# vLLM AR + graph_capture 句柄, _init_nccl 里初始化一次, 多次 case 复用.
_VLLM_TMP_AR = None
_VLLM_GRAPH_CAPTURE = None


def _torch_dtype(dtype_str: str):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]


def _init_nccl(local_rank: int) -> tuple[int, int]:
    """初始化 torch.distributed NCCL + vLLM distributed (供 AR 用真 vLLM 路径).

    AR 用 vllm.distributed.tensor_model_parallel_all_reduce, 它内部按 size 分发
    到 custom_all_reduce / pynccl / torch.dist (跟 production vLLM step 一致).
    裸 torch.dist.all_reduce 只覆盖 NCCL 一条, 小 size 跟生产路径偏差大.
    """
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    global _VLLM_TMP_AR, _VLLM_GRAPH_CAPTURE
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import (
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=world_size, rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank, backend="nccl",
    )
    # vLLM >= 0.14 要求 initialize_model_parallel() 在 VllmConfig context 内.
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        with set_current_vllm_config(VllmConfig()):
            initialize_model_parallel(
                tensor_model_parallel_size=world_size,
                pipeline_model_parallel_size=1,
            )
    except ImportError:
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
        )

    _VLLM_TMP_AR = tensor_model_parallel_all_reduce
    _VLLM_GRAPH_CAPTURE = graph_capture

    return rank, world_size


def _measure_vllm_ar_cudagraph(
    kernel_func, n_warmups: int = 3, n_iters: int = 20, repeat_n: int = 5,
) -> tuple[float, float, float]:
    """vLLM custom AR cudagraph 测量 (AIC 风格).

    vLLM TMP_AR 必须在 `vllm.graph_capture()` context 下 capture (custom_all_reduce
    + pynccl 依赖该 context 选 stream). 通用 harness 的 side-stream 路径不行.

    repeat_n 个 AR 串入同一 graph, replay×n_iters 取 p10/p50/p90, 单 AR latency
    = elapsed / repeat_n. 短 AR (<50us) 噪声明显降低.
    """
    import torch

    with _VLLM_GRAPH_CAPTURE(device=torch.cuda.current_device()) as gc_ctx:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gc_ctx.stream):
            for _ in range(repeat_n):
                kernel_func()

    torch.cuda.synchronize()
    for _ in range(n_warmups):
        graph.replay()
    torch.cuda.synchronize()

    latencies_us: list[float] = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1000.0 / repeat_n)

    latencies_us.sort()
    n = len(latencies_us)
    return (
        latencies_us[max(0, int(0.10 * n))],
        latencies_us[n // 2],
        latencies_us[min(n - 1, int(0.90 * n))],
    )


def _measure_vllm_ar_eager(
    kernel_func, n_warmups: int = 3, n_iters: int = 100, repeat_n: int = 5,
) -> tuple[float, float, float]:
    """vLLM custom AR eager 模式 (AIC 风格, 多次 repeat_n 串行).

    eager 模式 vLLM TMP_AR 不需要 graph_capture context. n_iters 比 cudagraph
    高 5× 抵消单次启动开销; 单 AR latency = elapsed / repeat_n.
    """
    import torch

    for _ in range(n_warmups):
        for _ in range(repeat_n):
            kernel_func()
    torch.cuda.synchronize()

    latencies_us: list[float] = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat_n):
            kernel_func()
        end.record()
        torch.cuda.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1000.0 / repeat_n)

    latencies_us.sort()
    n = len(latencies_us)
    return (
        latencies_us[max(0, int(0.10 * n))],
        latencies_us[n // 2],
        latencies_us[min(n - 1, int(0.90 * n))],
    )


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
    mode = params.get("execution_mode", "cudagraph")
    use_graph = (mode == "cudagraph")

    # barrier so all ranks reach measure together (avoid skew)
    dist.barrier()

    if op == "allreduce":
        # 走 vLLM 真路径 (TMP_AR 内部分发到 custom_all_reduce / pynccl / torch.dist).
        if _VLLM_TMP_AR is None:
            raise RuntimeError("vLLM AR not initialized; _init_nccl must run first")
        x = torch.randn(n_elems, dtype=dtype, device=device)

        def kernel_func() -> None:
            _VLLM_TMP_AR(x)

        if use_graph:
            p10, p50, p90 = _measure_vllm_ar_cudagraph(kernel_func)
        else:
            p10, p50, p90 = _measure_vllm_ar_eager(kernel_func)
        return (p10, p50, p90, use_graph, None)

    elif op == "alltoall":
        # vLLM 的 MoE EP dispatch/combine 调用栈太特殊 (DeepEP / pplx / naive),
        # 这一版先保留 torch.dist.all_to_all_single 做 baseline.
        if n_elems % world_size != 0:
            raise ValueError(
                f"alltoall n_elems={n_elems} not divisible by world_size={world_size}"
            )
        x = torch.randn(n_elems, dtype=dtype, device=device)
        out = torch.empty_like(x)

        def kernel_func() -> None:
            dist.all_to_all_single(out, x)

        cfg = (
            BenchConfig(n_warmups=3, n_iters=10, use_cuda_graph=True, allow_graph_fail=False)
            if use_graph
            else BenchConfig(n_warmups=3, n_iters=10, use_cuda_graph=False)
        )
        bench = measure(kernel_func, cfg)
        return (
            bench.latency_us_p10, bench.latency_us_p50, bench.latency_us_p90,
            bench.used_cuda_graph, bench.fallback_reason,
        )

    raise NotImplementedError(f"op_subtype {op!r} not supported")


_KERNEL_SOURCE_BY_OP = {
    # vLLM 的 TMP_AR 内部按 size 分发 (custom_all_reduce / pynccl / torch.dist),
    # 跟 production vLLM step 一致.
    "allreduce": "vllm_tensor_model_parallel_all_reduce",
    # alltoall 还是裸 torch.dist (vLLM EP path 这版未接).
    "alltoall": "torch_dist_nccl",
}


def _build_record(
    case: Case,
    p10: float, p50: float, p90: float,
    used_graph: bool,
    fallback_reason: Optional[str],
    *,
    framework_version: str,
    device_name: str,
) -> RawRecord:
    op = case.params.get("op_subtype", "unknown")
    # n_iters 跟实际测量函数对齐: vLLM AR cudagraph 用 20 (repeat_n=5), eager 用 100;
    # alltoall 走 harness 默认 10.
    if op == "allreduce":
        n_iters_used = 20 if used_graph else 100
    else:
        n_iters_used = 10
    return RawRecord(
        case_id=case.case_id,
        op_kind=OpKind.COLLECTIVE,
        framework=Framework.VLLM,
        framework_version=framework_version,
        device=device_name,
        execution_mode=(
            ExecutionMode.CUDAGRAPH if used_graph else ExecutionMode.EAGER
        ),
        kernel_source=_KERNEL_SOURCE_BY_OP.get(op, "torch_dist_nccl"),
        params=dict(case.params),
        metrics=Metrics(
            latency_us_p50=p50, latency_us_p10=p10, latency_us_p90=p90,
            used_cuda_graph=used_graph, n_warmups=3, n_iters=n_iters_used,
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
