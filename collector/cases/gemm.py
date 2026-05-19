"""GEMM cases — 从 ProfileSpec 派生 GEMM shape, 跨 profile dedup.

GEMM 种类(都是 [M, K] @ [K, N] → [M, N]):
  - qkv_proj:     K=hidden, N=qkv_out // tp                    (output-shard TP)
  - o_proj:       K=q_dim // tp, N=hidden                       (input-shard TP)
  - gate_up_proj: K=hidden, N=2 * intermediate // tp           (fused gate+up)
  - down_proj:    K=intermediate // tp, N=hidden
  - lm_head:      K=hidden, N=vocab                            (一般不 TP shard)

case_id 仅基于 (op_subtype, m, n, k, dtype, tp), profile_name 不进 hash.
"""
from __future__ import annotations

from collector.cases._dedup import merge_and_dedup
from collector.profiles import ProfileSpec
from collector.schemas import Case, OpKind


# 默认 sweep 维度. CLI 可 override.
DEFAULT_M_VALUES: list[int] = [1, 4, 16, 32, 128, 512, 2048, 4096, 8192]
DEFAULT_TP_SIZES: list[int] = [1]
DEFAULT_DTYPES: list[str] = ["bf16"]
# 跟 vLLM production 部署对齐: --enforce-eager (eager) 跟 V1 default (cudagraph) 两路.
DEFAULT_EXECUTION_MODES: list[str] = ["eager", "cudagraph"]


def get_cases_for_profile(
    profile: ProfileSpec,
    *,
    m_values: list[int] | None = None,
    tp_sizes: list[int] | None = None,
    dtypes: list[str] | None = None,
    execution_modes: list[str] | None = None,
    include_mlp: bool | None = None,
    include_lm_head: bool = True,
) -> list[Case]:
    """从单个 profile 派生 GEMM cases.

    Args:
        execution_modes: 每个 shape 在哪些模式下采 (eager/cudagraph), 进 case_id hash.
        include_mlp: None → 自动 (profile 有 dense FFN 且非 MoE 时 True)
                     True/False → 强制 (例 MoE-only 模型设 False)
    """
    m_values = m_values or DEFAULT_M_VALUES
    tp_sizes = tp_sizes or DEFAULT_TP_SIZES
    dtypes = dtypes or DEFAULT_DTYPES
    execution_modes = execution_modes or DEFAULT_EXECUTION_MODES
    if include_mlp is None:
        # 默认: MoE 模型不含 dense FFN GEMM (它们走 fused_moe op)
        include_mlp = not profile.has_moe and profile.intermediate > 0

    cases: list[Case] = []

    for mode in execution_modes:
        for tp in tp_sizes:
            for dtype in dtypes:
                for m in m_values:
                    cases.append(_gemm_case(
                        "qkv_proj",
                        m=m, k=profile.hidden, n=profile.qkv_out // tp,
                        dtype=dtype, tp=tp, execution_mode=mode,
                    ))
                    cases.append(_gemm_case(
                        "o_proj",
                        m=m, k=profile.q_dim // tp, n=profile.hidden,
                        dtype=dtype, tp=tp, execution_mode=mode,
                    ))
                    if include_mlp:
                        cases.append(_gemm_case(
                            "gate_up_proj",
                            m=m, k=profile.hidden, n=2 * profile.intermediate // tp,
                            dtype=dtype, tp=tp, execution_mode=mode,
                        ))
                        cases.append(_gemm_case(
                            "down_proj",
                            m=m, k=profile.intermediate // tp, n=profile.hidden,
                            dtype=dtype, tp=tp, execution_mode=mode,
                        ))

    if include_lm_head:
        for mode in execution_modes:
            for dtype in dtypes:
                for m in m_values:
                    cases.append(_gemm_case(
                        "lm_head",
                        m=m, k=profile.hidden, n=profile.vocab,
                        dtype=dtype, tp=1, execution_mode=mode,
                    ))

    return cases


def get_cases(
    profiles: list[ProfileSpec],
    **opts,
) -> tuple[list[Case], dict[str, list[str]]]:
    """多 profile case 合并 + dedup.

    Returns:
        (unique_cases, source_profiles_by_case_id) — provenance 走 metadata, 不进 case
    """
    per_profile = [
        (p.profile_name, get_cases_for_profile(p, **opts))
        for p in profiles
    ]
    return merge_and_dedup(per_profile)


def _gemm_case(op_subtype: str,
               *, m: int, n: int, k: int, dtype: str, tp: int,
               execution_mode: str) -> Case:
    """构造单个 GEMM Case. case_id = hash(params), 不含 profile_name."""
    return Case.make(
        OpKind.GEMM,
        params={
            "op_subtype": op_subtype,
            "m": m,
            "n": n,
            "k": k,
            "dtype": dtype,
            "tp": tp,
            "execution_mode": execution_mode,
        },
        prefix=f"{op_subtype}_tp{tp}_m{m}_{execution_mode}",
    )
