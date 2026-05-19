"""MoE cases — 从 has_moe 的 ProfileSpec 派生 fused MoE shape × routing distribution.

vLLM `fused_moe` kernel 一次调用处理 [num_tokens, topk, num_experts] 的全部 expert
路由 + 计算. 关键性能轴是 num_tokens 跟 routing distribution (token 到 expert 的
分布):

  balanced            - 每 expert 收到 ~mean(num_tokens × topk / num_experts) token
                        理想 case, training load-balance loss 推这个方向
  power_law(α=1.01)   - 轻度 skew, 实际 production 弱倾斜场景
  power_law(α=1.2)    - 强 skew, hot expert 收 token 远超 mean

EP>1 时, 最忙 rank 决定 step time; balanced 跟 power_law 差异在 EP path 上放大.
TP>1 时, 每 expert weight 沿 intermediate 维 shard, fused_moe 内部隐式 AllReduce.

参考 AIC `collector/common_test_cases.py:get_common_moe_test_cases`
跟 `collector/helper.py:balanced_logits / power_law_logits_v3`.
"""
from __future__ import annotations

from collector.cases._dedup import merge_and_dedup
from collector.profiles import ProfileSpec
from collector.schemas import Case, OpKind


# ---------------------------------------------------------------------------
# Sweep defaults — 跟 GEMM/attention 保持 M-值一致, 加 MoE 特有 routing 维度
# ---------------------------------------------------------------------------

DEFAULT_NUM_TOKENS: list[int] = [1, 4, 16, 32, 128, 512, 2048, 4096, 8192]

# (tp, ep) 配对. Stage M-A/B bench 用过 (4, 1) (AllReduce path) 跟 (4, 4) (AllToAll path).
# (1, 1) 是单 GPU baseline, (1, 4) 测纯 EP 路径.
DEFAULT_PARALLEL: list[tuple[int, int]] = [(1, 1), (1, 4), (4, 1), (4, 4)]

# Routing: 跟 AIC 完全对齐 (3 档).
# (distribution, alpha) — alpha=0.0 对 balanced 是占位.
DEFAULT_ROUTINGS: list[tuple[str, float]] = [
    ("balanced", 0.0),
    ("power_law", 1.01),
    ("power_law", 1.2),
]

DEFAULT_DTYPES: list[str] = ["bf16"]
DEFAULT_EXECUTION_MODES: list[str] = ["eager", "cudagraph"]


# ---------------------------------------------------------------------------
# Single profile
# ---------------------------------------------------------------------------

def get_cases_for_profile(
    profile: ProfileSpec,
    *,
    num_tokens_values: list[int] | None = None,
    parallel_configs: list[tuple[int, int]] | None = None,
    routings: list[tuple[str, float]] | None = None,
    dtypes: list[str] | None = None,
    execution_modes: list[str] | None = None,
) -> list[Case]:
    """从单个 profile 派生 MoE cases. 非 MoE profile 返空 list."""
    if not profile.has_moe:
        return []

    num_tokens_values = num_tokens_values or DEFAULT_NUM_TOKENS
    parallel_configs = parallel_configs or DEFAULT_PARALLEL
    routings = routings or DEFAULT_ROUTINGS
    dtypes = dtypes or DEFAULT_DTYPES
    execution_modes = execution_modes or DEFAULT_EXECUTION_MODES

    cases: list[Case] = []

    for mode in execution_modes:
        for tp, ep in parallel_configs:
            if ep > profile.moe_num_experts:
                continue
            for routing, alpha in routings:
                for dtype in dtypes:
                    for n in num_tokens_values:
                        cases.append(_moe_case(
                            num_tokens=n,
                            hidden=profile.hidden,
                            moe_intermediate=profile.moe_intermediate,
                            topk=profile.moe_top_k,
                            num_experts=profile.moe_num_experts,
                            tp=tp, ep=ep,
                            routing_distribution=routing,
                            power_law_alpha=alpha,
                            dtype=dtype,
                            execution_mode=mode,
                        ))

    return cases


# ---------------------------------------------------------------------------
# Multi-profile dedup
# ---------------------------------------------------------------------------

def get_cases(
    profiles: list[ProfileSpec],
    **opts,
) -> tuple[list[Case], dict[str, list[str]]]:
    """跨多 MoE profile 合并 + dedup by case_id.

    非 MoE profile 自动跳过. 同 MoE shape 跨 profile 自然 dedup.
    """
    per_profile = [
        (p.profile_name, get_cases_for_profile(p, **opts))
        for p in profiles
    ]
    return merge_and_dedup(per_profile)


# ---------------------------------------------------------------------------
# Case constructor
# ---------------------------------------------------------------------------

def _moe_case(
    *,
    num_tokens: int,
    hidden: int,
    moe_intermediate: int,
    topk: int,
    num_experts: int,
    tp: int,
    ep: int,
    routing_distribution: str,
    power_law_alpha: float,
    dtype: str,
    execution_mode: str,
) -> Case:
    return Case.make(
        OpKind.MOE,
        params={
            "num_tokens": num_tokens,
            "hidden": hidden,
            "moe_intermediate": moe_intermediate,
            "topk": topk,
            "num_experts": num_experts,
            "tp": tp,
            "ep": ep,
            "routing_distribution": routing_distribution,
            "power_law_alpha": power_law_alpha,
            "dtype": dtype,
            "execution_mode": execution_mode,
        },
        prefix=_moe_prefix(num_tokens, tp, ep, routing_distribution,
                           power_law_alpha, execution_mode),
    )


def _moe_prefix(num_tokens: int, tp: int, ep: int,
                routing: str, alpha: float, execution_mode: str) -> str:
    rt_tag = "balanced" if routing == "balanced" else f"{routing}_{alpha}"
    return f"moe_n{num_tokens}_tp{tp}_ep{ep}_{rt_tag}_{execution_mode}"
