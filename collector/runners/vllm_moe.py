"""vLLM MoE runner — 测一个 rank 的 fused_experts 本地 compute (单 GPU).

任意 (tp, ep) 都在单 GPU 上 sim:
    local_inter   = moe_intermediate // tp
    local_experts = num_experts // ep
    expert_map    = [-1 if global_id not in this_rank's slice else local_idx]

跟 AIC `collector/vllm/collect_moe_v2.py` 同思路: 单 GPU compute,
通信 (AllReduce for TP / AllToAll for EP) 由 collective collector 单独测,
sim cost model 自己加. 这样 op 数据语义干净:
    MoE 数据    = 一个 rank 的 fused_experts compute (无 dist)
    collective 数据 = AllReduce / AllToAll 单独

Routing:
    balanced              - 每 expert 收到 ~mean(M × topk / num_experts)
    power_law(α)          - 按 rank^(-α) 偏置, 少数 expert 拿走大头
跟 case.params 字段对齐 (cases/moe.py).

BF16 unquantized 第一版.
"""
from __future__ import annotations

from collector.harness import BenchConfig, BenchResult, measure
from collector.runners._vllm_dist import ensure_initialized, torch_dtype
from collector.schemas import (
    Case,
    ExecutionMode,
    Framework,
    Metrics,
    OpKind,
    RawRecord,
)


def build_record(
    case: Case,
    bench: BenchResult,
    *,
    framework_version: str,
    device_name: str,
    kernel_source: str = "vllm_fused_experts",
) -> RawRecord:
    return RawRecord(
        case_id=case.case_id,
        op_kind=OpKind.MOE,
        framework=Framework.VLLM,
        framework_version=framework_version,
        device=device_name,
        execution_mode=(
            ExecutionMode.CUDAGRAPH if bench.used_cuda_graph else ExecutionMode.EAGER
        ),
        kernel_source=kernel_source,
        params=dict(case.params),
        metrics=Metrics(
            latency_us_p50=bench.latency_us_p50,
            latency_us_p10=bench.latency_us_p10,
            latency_us_p90=bench.latency_us_p90,
            used_cuda_graph=bench.used_cuda_graph,
            n_warmups=bench.n_warmups,
            n_iters=bench.n_iters,
        ),
        metadata={
            "fallback_reason": bench.fallback_reason,
        },
    )


def _build_routing(
    num_tokens: int,
    topk: int,
    num_experts: int,
    distribution: str,
    alpha: float,
    device: str,
):
    """构造 (topk_weights, topk_ids) 模拟指定分布. 返 (M, topk) shape."""
    import torch

    if distribution == "balanced":
        # round-robin: 每 token 的 topk 个槽顺序覆盖所有 expert,
        # 起点跟着 token id 偏一下避免 row 之间完全一样.
        ids = torch.arange(num_tokens * topk, device=device, dtype=torch.int32)
        topk_ids = (ids % num_experts).view(num_tokens, topk).to(torch.int32)
    elif distribution == "power_law":
        # rank r ∈ [1..E], P(r) ∝ r^(-alpha). 采样后映射到随机 expert permutation,
        # 保证每次 case 一致但 expert 顺序不偏 expert id=0.
        ranks = torch.arange(1, num_experts + 1, dtype=torch.float64)
        probs = ranks ** (-alpha)
        probs = probs / probs.sum()
        gen = torch.Generator(device="cpu").manual_seed(0xA15)
        sampled = torch.multinomial(
            probs, num_samples=num_tokens * topk, replacement=True, generator=gen,
        ).to(torch.int32)
        # 把 rank → expert id permutation, 避免 expert 0 永远是热门
        perm = torch.randperm(num_experts, generator=gen).to(torch.int32)
        topk_ids = perm[sampled].view(num_tokens, topk).to(device).to(torch.int32)
    else:
        raise NotImplementedError(f"routing distribution {distribution!r} not supported")

    # topk_weights: 简单 uniform 1/topk, fused_experts 不依赖具体值 (仅做加权和)
    topk_weights = torch.full(
        (num_tokens, topk), 1.0 / topk, device=device, dtype=torch.float32,
    )
    return topk_weights, topk_ids


def run_case(case: Case, device: int) -> RawRecord:
    """跑单 MoE case (任意 tp, ep, 单 GPU sim 一个 rank 的本地 compute).

    case.params 必含: num_tokens, hidden, moe_intermediate, topk, num_experts,
                     tp, ep, routing_distribution, power_law_alpha, dtype, execution_mode
    """
    p = case.params

    if p["dtype"] != "bf16":
        raise NotImplementedError(
            f"vllm_moe runner: dtype={p['dtype']} not supported (BF16 only)"
        )

    tp = int(p.get("tp", 1))
    ep = int(p.get("ep", 1))

    import torch
    import vllm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_experts

    dtype = torch_dtype(p["dtype"])
    device_str = f"cuda:{device}"

    M = int(p["num_tokens"])
    hidden = int(p["hidden"])
    moe_inter = int(p["moe_intermediate"])
    topk = int(p["topk"])
    E_global = int(p["num_experts"])

    if moe_inter % tp != 0:
        raise NotImplementedError(
            f"moe_intermediate={moe_inter} not divisible by tp={tp}"
        )
    if E_global % ep != 0:
        raise NotImplementedError(
            f"num_experts={E_global} not divisible by ep={ep}"
        )

    local_inter = moe_inter // tp
    E_local = E_global // ep

    # 一个 rank 持有: [E/ep experts, 2 × N/tp gate+up, hidden] / [E/ep, hidden, N/tp]
    w1_shape = (E_local, 2 * local_inter, hidden)
    w2_shape = (E_local, hidden, local_inter)

    with set_current_vllm_config(VllmConfig()):
        ensure_initialized(device)

        x = torch.randn((M, hidden), dtype=dtype, device=device_str)
        w1 = torch.randn(w1_shape, dtype=dtype, device=device_str)
        w2 = torch.randn(w2_shape, dtype=dtype, device=device_str)
        topk_weights, topk_ids = _build_routing(
            M, topk, E_global,
            p["routing_distribution"], float(p["power_law_alpha"]),
            device_str,
        )

        # expert_map: global expert id → local idx, 不在本 rank 的 expert 标 -1
        # ep=1 时全 expert 都在本地, 不需要 expert_map.
        if ep > 1:
            expert_map = torch.full(
                (E_global,), -1, dtype=torch.int32, device=device_str,
            )
            # rank 0 的 ep_rank=0 → 持 expert [0, E_local)
            expert_map[:E_local] = torch.arange(
                E_local, dtype=torch.int32, device=device_str,
            )
        else:
            expert_map = None

        def kernel_func() -> None:
            fused_experts(
                x, w1, w2, topk_weights, topk_ids,
                global_num_experts=E_global,
                expert_map=expert_map,
            )

        # dry run (Triton autotune + kernel selection)
        kernel_func()

        mode = p.get("execution_mode", "cudagraph")
        if mode == "eager":
            cfg = BenchConfig(n_warmups=3, n_iters=10, use_cuda_graph=False)
        elif mode == "cudagraph":
            cfg = BenchConfig(n_warmups=3, n_iters=10,
                              use_cuda_graph=True, allow_graph_fail=False)
        else:
            raise NotImplementedError(f"execution_mode {mode!r} not supported")
        bench = measure(kernel_func, cfg)

    device_name = torch.cuda.get_device_name(device)
    return build_record(
        case, bench,
        framework_version=str(vllm.__version__),
        device_name=device_name,
    )
