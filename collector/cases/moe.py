"""MoE cases — AIC-aligned raw params (moe_plan Phase 2).

vLLM `fused_moe` kernel 一次调用处理 [num_tokens, topk, num_experts] 的全部 expert
路由 + 计算. 关键性能轴是 num_tokens 跟 routing distribution (token 到 expert 的
分布):

  balanced            - 每 expert 收到 ~mean(num_tokens × topk / num_experts) token
                        理想 case, training load-balance loss 推这个方向
  power_law_1.01      - 轻度 skew, 实际 production 弱倾斜场景
  power_law_1.2       - 强 skew, hot expert 收 token 远超 mean

EP>1 时, 最忙 rank 决定 step time; balanced 跟 power_law 差异在 EP path 上放大.
TP>1 时, 每 expert weight 沿 intermediate 维 shard, fused_moe 内部隐式 AllReduce.

字段口径 (moe_plan §3.6.1, AIC-aligned):
    moe_dtype       - bfloat16 / float16 (default bf16; RTX 4090 production 用 bf16)
    num_tokens      - input token count
    hidden_size     - model hidden (AIC 命名, 不再是 internal 'hidden')
    inter_size      - per-expert intermediate, AIC 命名 (不再是 'moe_intermediate')
    topk            - activated experts per token
    num_experts     - global routed expert count
    moe_tp_size     - MoE expert TP size (AIC; 内部 alias 仍是 'tp')
    moe_ep_size     - MoE expert EP size (AIC; 内部 alias 仍是 'ep')
    distribution    - 'balanced' / 'power_law_1.01' / 'power_law_1.2'
                      (AIC 单字段编码; 内部 canonical 拆 routing_distribution +
                       power_law_alpha 由 moe_case_params_to_signature 转)
    execution_mode  - cudagraph / eager

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

DEFAULT_NUM_TOKENS: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]

# (moe_tp_size, moe_ep_size) 配对. moe_plan §4 Phase 2 step 4 限定本轮 vLLM 最小集:
#   (4, 1)  TP-only routed allreduce path
#   (1, 4)  EP-only path (vLLM 不支持 moe_tp_size>1 AND moe_ep_size>1)
DEFAULT_PARALLEL: list[tuple[int, int]] = [(4, 1), (1, 4)]

# AIC 单字段编码 (moe_plan §3.6.1):
DEFAULT_DISTRIBUTIONS: list[str] = ["balanced", "power_law_1.01", "power_law_1.2"]

# moe_plan Phase 2 step 1: 默认 bfloat16 (RTX 4090 production), float16 作可选对照
DEFAULT_MOE_DTYPES: list[str] = ["bfloat16"]

DEFAULT_EXECUTION_MODES: list[str] = ["cudagraph"]


# ---------------------------------------------------------------------------
# Single profile
# ---------------------------------------------------------------------------

def get_cases_for_profile(
    profile: ProfileSpec,
    *,
    num_tokens_values: list[int] | None = None,
    parallel_configs: list[tuple[int, int]] | None = None,
    distributions: list[str] | None = None,
    moe_dtypes: list[str] | None = None,
    execution_modes: list[str] | None = None,
) -> list[Case]:
    """从单个 profile 派生 MoE cases. 非 MoE profile 返空 list."""
    if not profile.has_moe:
        return []

    num_tokens_values = num_tokens_values or DEFAULT_NUM_TOKENS
    parallel_configs = parallel_configs or DEFAULT_PARALLEL
    distributions = distributions or DEFAULT_DISTRIBUTIONS
    moe_dtypes = moe_dtypes or DEFAULT_MOE_DTYPES
    execution_modes = execution_modes or DEFAULT_EXECUTION_MODES

    cases: list[Case] = []

    for mode in execution_modes:
        for moe_tp, moe_ep in parallel_configs:
            if moe_ep > profile.moe_num_experts:
                continue
            if moe_tp > 1 and moe_ep > 1:
                # vLLM 不支持 moe_tp_size>1 AND moe_ep_size>1 同时, plan Phase 2 step 4 注释
                continue
            for dist in distributions:
                for moe_dtype in moe_dtypes:
                    for n in num_tokens_values:
                        cases.append(_moe_case(
                            num_tokens=n,
                            hidden_size=profile.hidden,
                            inter_size=profile.moe_intermediate,
                            topk=profile.moe_top_k,
                            num_experts=profile.moe_num_experts,
                            moe_tp_size=moe_tp,
                            moe_ep_size=moe_ep,
                            distribution=dist,
                            moe_dtype=moe_dtype,
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
# Case constructor (AIC-aligned)
# ---------------------------------------------------------------------------

def _moe_case(
    *,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    distribution: str,
    moe_dtype: str,
    execution_mode: str,
) -> Case:
    return Case.make(
        OpKind.MOE,
        params={
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "inter_size": inter_size,
            "topk": topk,
            "num_experts": num_experts,
            "moe_tp_size": moe_tp_size,
            "moe_ep_size": moe_ep_size,
            "distribution": distribution,
            "moe_dtype": moe_dtype,
            "execution_mode": execution_mode,
        },
        prefix=_moe_prefix(
            num_tokens, moe_tp_size, moe_ep_size, distribution, execution_mode,
        ),
    )


def _moe_prefix(
    num_tokens: int, moe_tp: int, moe_ep: int,
    distribution: str, execution_mode: str,
) -> str:
    return (
        f"moe_n{num_tokens}_motp{moe_tp}_moep{moe_ep}"
        f"_{distribution}_{execution_mode}"
    )
