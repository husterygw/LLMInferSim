"""vLLM MoE runner — 测一个 rank 的 fused_experts 本地 compute (单 GPU).

moe_plan Phase 2: 字段口径 AIC-aligned (raw record 直接产 AIC 字段).
    moe_dtype       - bfloat16 / float16 (默认 bf16)
    num_tokens
    hidden_size     - 替代 internal 'hidden'
    inter_size      - 替代 internal 'moe_intermediate'
    topk
    num_experts
    moe_tp_size     - 替代 internal 'tp'
    moe_ep_size     - 替代 internal 'ep'
    distribution    - 'balanced' / 'power_law_<alpha>'
    execution_mode  - cudagraph / eager

任意合法 (moe_tp, moe_ep) 组合都在单 GPU 上 sim 一个 rank 的本地 compute.
vLLM 不支持 moe_tp_size>1 AND moe_ep_size>1 同时启用 (TP-on-intermediate 跟
EP-on-experts 两个 sharding 维度互斥), 该组合由 cases/moe.py:get_cases_for_profile
skip + runner 入口 assert 双重兜底防误用. 合法组合下:

    ep=1, tp>1: local_inter = inter_size // moe_tp   (TP 沿 intermediate 切)
                E_local     = num_experts            (全 expert 在本卡)
                expert_map  = None

    ep>1, tp=1: local_inter = inter_size             (intermediate 不切, 整 expert)
                E_local     = num_experts // moe_ep  (EP 切 expert 集合)
                expert_map  = vLLM determine_expert_map(ep_size=moe_ep, ep_rank=0, ...)

跟 AIC `aiconfigurator/collector/vllm/collect_moe_v2.py` 同思路: 单 GPU compute,
通信 (AllReduce for TP / AllToAll for EP) 由 collective collector 单独测,
sim cost model 自己加. 数据语义干净:
    MoE 数据    = 一个 rank 的 fused_experts compute (无 dist)
    collective 数据 = AllReduce / AllToAll 单独

Routing logits 生成跟 AIC 对齐 (`aiconfigurator/collector/helper.py` 内嵌简化版):
    balanced              - balanced_logits + topk → (topk_weights, topk_ids)
    power_law_<alpha>     - power_law_logits + topk
"""
from __future__ import annotations

from collector.harness import BenchConfig, BenchResult, measure
from collector.runners._vllm_dist import ensure_initialized
from collector.schemas import (
    Case,
    ExecutionMode,
    Framework,
    Metrics,
    OpKind,
    RawRecord,
)


# ---------------------------------------------------------------------------
# Dtype: AIC moe_dtype (bfloat16/float16) → torch dtype
# (不复用 collector/runners/_vllm_dist.torch_dtype 因为它读 internal 'bf16'/'fp16' 命名)
# ---------------------------------------------------------------------------

def _moe_dtype_to_torch(moe_dtype: str):
    import torch
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if moe_dtype not in table:
        raise NotImplementedError(
            f"moe_dtype={moe_dtype!r} not supported "
            f"(Phase 2 仅 bfloat16/float16; int4_wo/fp8/fp4/mxfp4 后续 phase)"
        )
    return table[moe_dtype]


# ---------------------------------------------------------------------------
# AIC-aligned routing logits (内嵌简化版, 跟 aiconfigurator/collector/helper.py
# balanced_logits / power_law_logits_v3 数学等价, 不依赖 EPLB / WideEP / num_slots)
# ---------------------------------------------------------------------------

def _balanced_logits(num_tokens: int, num_experts: int, topk: int, device: str):
    """AIC balanced_logits 复刻: 返 (num_tokens, num_experts) softmax logits."""
    import math
    import torch
    import torch.nn.functional as F

    stride = math.ceil(num_experts / topk)
    token_indices = torch.arange(num_tokens).unsqueeze(1)
    topk_indices = torch.arange(topk).unsqueeze(0)
    if num_tokens >= stride:
        h_selected = (token_indices + topk_indices * stride) % num_experts
    else:
        h_selected = (
            (token_indices * stride / num_tokens + topk_indices * stride).long()
            % num_experts
        )
    expert_one_hot = F.one_hot(h_selected.long(), num_classes=num_experts).sum(1)
    return F.softmax(expert_one_hot.to(torch.bfloat16), dim=1).to(device)


def _power_law_logits(
    num_tokens: int, num_experts: int, topk: int, alpha: float,
    device: str, seed: int = 0xA15,
):
    """AIC power_law_logits_v3 复刻 (无 EPLB / WideEP / num_slots 路径).

    rank r ∈ [1..E], P(r) ∝ r^(-alpha). 采样后通过 expert permutation 打乱
    rank→expert 映射, 避免 expert id=0 永远是热门. fixed seed 保证 case_id
    跟 latency 1:1 可重现.
    """
    import torch
    import torch.nn.functional as F

    ranks = torch.arange(1, num_experts + 1, dtype=torch.float64)
    probs = ranks ** (-alpha)
    probs = probs / probs.sum()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sampled = torch.multinomial(
        probs, num_samples=num_tokens * topk,
        replacement=True, generator=gen,
    ).reshape(num_tokens, topk)
    perm = torch.randperm(num_experts, generator=gen)
    sampled = perm[sampled]
    expert_one_hot = F.one_hot(sampled.long(), num_classes=num_experts).sum(1)
    return F.softmax(expert_one_hot.to(torch.bfloat16), dim=1).to(device)


def _build_routing(
    num_tokens: int, topk: int, num_experts: int,
    distribution: str, device: str,
):
    """生成 (topk_weights, topk_ids) 给 fused_experts.

    distribution 解码:
        'balanced'         → balanced_logits + topk
        'power_law_<alpha>'→ power_law_logits(alpha=<alpha>) + topk
    """
    import torch
    import torch.nn.functional as F

    if distribution == "balanced":
        logits = _balanced_logits(num_tokens, num_experts, topk, device)
    elif distribution.startswith("power_law_"):
        try:
            alpha = float(distribution[len("power_law_"):])
        except ValueError as e:
            raise ValueError(
                f"distribution={distribution!r} 不是有效 power_law_<alpha> 格式"
            ) from e
        logits = _power_law_logits(num_tokens, num_experts, topk, alpha, device)
    else:
        raise NotImplementedError(
            f"distribution={distribution!r} not supported "
            f"(仅 balanced / power_law_<alpha>)"
        )

    weights, ids = torch.topk(logits, topk, dim=-1)
    topk_weights = F.softmax(weights, dim=-1).to(torch.float32)
    topk_ids = ids.to(torch.int32)
    return topk_weights, topk_ids


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def build_record(
    case: Case,
    bench: BenchResult,
    *,
    framework_version: str,
    device_name: str,
    kernel_source: str = "vllm_fused_moe",
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


# ---------------------------------------------------------------------------
# run_case
# ---------------------------------------------------------------------------

def run_case(case: Case, device: int) -> RawRecord:
    """跑单 MoE case (单 GPU sim 一个 rank 的本地 compute).

    case.params 必含 (AIC-aligned, moe_plan §3.6.1):
        num_tokens, hidden_size, inter_size, topk, num_experts,
        moe_tp_size, moe_ep_size, distribution, moe_dtype, execution_mode
    """
    p = case.params

    import torch
    import vllm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.layer import determine_expert_map

    dtype = _moe_dtype_to_torch(p["moe_dtype"])
    device_str = f"cuda:{device}"

    M = int(p["num_tokens"])
    hidden = int(p["hidden_size"])
    moe_inter = int(p["inter_size"])
    topk = int(p["topk"])
    E_global = int(p["num_experts"])
    moe_tp = int(p["moe_tp_size"])
    moe_ep = int(p["moe_ep_size"])
    distribution = str(p["distribution"])

    # vLLM 不支持 TP-on-intermediate 跟 EP-on-experts 两个 sharding 维度同时启用.
    # cases/moe.py:get_cases_for_profile 已 skip 该组合, 这里做 belt-and-suspenders 兜底
    # 防止后续手写 case 直接调 runner 跑出语义错误数据.
    if moe_tp > 1 and moe_ep > 1:
        raise NotImplementedError(
            f"moe_tp_size={moe_tp} AND moe_ep_size={moe_ep}: vLLM 不支持 TP+EP 同时启用 "
            f"(TP 沿 intermediate / EP 沿 experts 两个 sharding 维度互斥, "
            f"cases/moe.py 应已 skip)"
        )
    if moe_inter % moe_tp != 0:
        raise NotImplementedError(
            f"inter_size={moe_inter} not divisible by moe_tp_size={moe_tp}"
        )
    if E_global % moe_ep != 0:
        raise NotImplementedError(
            f"num_experts={E_global} not divisible by moe_ep_size={moe_ep}"
        )

    local_inter = moe_inter // moe_tp
    E_local = E_global // moe_ep

    w1_shape = (E_local, 2 * local_inter, hidden)
    w2_shape = (E_local, hidden, local_inter)

    with set_current_vllm_config(VllmConfig()):
        ensure_initialized(device)

        x = torch.randn((M, hidden), dtype=dtype, device=device_str)
        w1 = torch.randn(w1_shape, dtype=dtype, device=device_str)
        w2 = torch.randn(w2_shape, dtype=dtype, device=device_str)
        topk_weights, topk_ids = _build_routing(
            M, topk, E_global, distribution, device_str,
        )

        # expert_map: vLLM determine_expert_map (ep_rank=0).
        # ep=1 → expert_map=None (全 expert local).
        if moe_ep > 1:
            _, expert_map, _ = determine_expert_map(
                ep_size=moe_ep, ep_rank=0, global_num_experts=E_global,
            )
            if expert_map is not None:
                expert_map = expert_map.to(device_str)
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
            cfg = BenchConfig(
                n_warmups=3, n_iters=10,
                use_cuda_graph=True, allow_graph_fail=False,
            )
        else:
            raise NotImplementedError(f"execution_mode {mode!r} not supported")
        bench = measure(kernel_func, cfg)

    device_name = torch.cuda.get_device_name(device)
    return build_record(
        case, bench,
        framework_version=str(vllm.__version__),
        device_name=device_name,
    )
