"""Attention cases — 4D batch shape, 从 ProfileSpec 派生.

case_id 仅基于 (phase, batch, isl, kv_prefill, n_decode, kv_decode,
num_heads, num_kv_heads, head_dim, dtype, tp); profile_name 不进 hash.
"""
from __future__ import annotations

from collector.cases._dedup import merge_and_dedup
from collector.profiles import ProfileSpec
from collector.schemas import Case, OpKind


DEFAULT_PREFILL_ISLS: list[int] = [128, 512, 2048, 4096, 8192]
DEFAULT_DECODE_BATCHES: list[int] = [1, 4, 16, 32]
DEFAULT_DECODE_CTX_LENS: list[int] = [128, 512, 2048, 4096, 8192]
DEFAULT_TP_SIZES: list[int] = [1]
DEFAULT_DTYPES: list[str] = ["bf16"]
DEFAULT_EXECUTION_MODES: list[str] = ["eager", "cudagraph"]


def get_cases_for_profile(
    profile: ProfileSpec,
    *,
    prefill_isls: list[int] | None = None,
    decode_batches: list[int] | None = None,
    decode_ctx_lens: list[int] | None = None,
    tp_sizes: list[int] | None = None,
    dtypes: list[str] | None = None,
    execution_modes: list[str] | None = None,
    include_prefill: bool = True,
    include_decode: bool = True,
) -> list[Case]:
    prefill_isls = prefill_isls or DEFAULT_PREFILL_ISLS
    decode_batches = decode_batches or DEFAULT_DECODE_BATCHES
    decode_ctx_lens = decode_ctx_lens or DEFAULT_DECODE_CTX_LENS
    tp_sizes = tp_sizes or DEFAULT_TP_SIZES
    dtypes = dtypes or DEFAULT_DTYPES
    execution_modes = execution_modes or DEFAULT_EXECUTION_MODES

    head_info = {
        "num_heads": profile.num_heads,
        "num_kv_heads": profile.num_kv_heads,
        "head_dim": profile.head_dim,
    }
    cases: list[Case] = []

    if include_prefill:
        for mode in execution_modes:
            for tp in tp_sizes:
                for dtype in dtypes:
                    for isl in prefill_isls:
                        cases.append(_attention_case(
                            "prefill",
                            batch_size=1, isl=isl, kv_prefill=0,
                            n_decode=0, kv_decode=0,
                            dtype=dtype, tp=tp, head_info=head_info,
                            execution_mode=mode,
                        ))

    if include_decode:
        for mode in execution_modes:
            for tp in tp_sizes:
                for dtype in dtypes:
                    for batch in decode_batches:
                        for ctx in decode_ctx_lens:
                            cases.append(_attention_case(
                                "decode",
                                batch_size=batch, isl=0, kv_prefill=0,
                                n_decode=batch, kv_decode=ctx,
                                dtype=dtype, tp=tp, head_info=head_info,
                                execution_mode=mode,
                            ))

    return cases


def get_cases(
    profiles: list[ProfileSpec],
    **opts,
) -> tuple[list[Case], dict[str, list[str]]]:
    per_profile = [
        (p.profile_name, get_cases_for_profile(p, **opts))
        for p in profiles
    ]
    return merge_and_dedup(per_profile)


def _attention_case(
    phase: str,
    *, batch_size: int, isl: int, kv_prefill: int,
    n_decode: int, kv_decode: int,
    dtype: str, tp: int, head_info: dict,
    execution_mode: str,
) -> Case:
    return Case.make(
        OpKind.ATTENTION,
        params={
            "phase": phase,
            "batch_size": batch_size,
            "isl": isl,
            "kv_prefill": kv_prefill,
            "n_decode": n_decode,
            "kv_decode": kv_decode,
            "num_heads": head_info["num_heads"],
            "num_kv_heads": head_info["num_kv_heads"],
            "head_dim": head_info["head_dim"],
            "dtype": dtype,
            "tp": tp,
            "execution_mode": execution_mode,
        },
        prefix=(
            f"prefill_isl{isl}_tp{tp}_{execution_mode}"
            if phase == "prefill"
            else f"decode_b{batch_size}_ctx{kv_decode}_tp{tp}_{execution_mode}"
        ),
    )
