"""Attention canonicalizer — V3 §5.3 / IMPL_PLAN §2.5.

字段:
    op_kind   = attention
    op_subtype= prefill / decode / mixed_split / mixed_unified (Stage 2: 仅 prefill / decode)
    dtype
    shape     = {num_tokens, num_seqs, q_len, kv_len, num_q_heads, num_kv_heads, head_dim}
    parallel  = {tp}
    runtime   = {framework, framework_version, execution_mode, kernel_source,
                 attention_backend, kv_dtype, block_size}

Mapping collector phase 'prefill' / 'decode' → runtime shape:
    prefill: num_seqs=batch_size, num_tokens=batch_size*isl, q_len=isl, kv_len=isl
    decode : num_seqs=n_decode,   num_tokens=n_decode,        q_len=1,   kv_len=kv_decode
"""
from __future__ import annotations

from typing import Any

from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature

_SHAPE_KEYS = (
    "num_tokens", "num_seqs", "q_len", "kv_len",
    # mixed (unified prefill+decode kernel) 段字段; 纯 prefill/decode 不含 → project
    # 填 None → to_canonical 跳过 → 纯 prefill/decode hash 与旧版兼容。
    "prefill_chunk", "n_prefill", "kv_prefill", "n_decode", "kv_decode",
    "num_q_heads", "num_kv_heads", "head_dim",
)
_PARALLEL_KEYS = ("tp",)
_RUNTIME_KEYS = (
    "framework", "framework_version", "execution_mode", "kernel_source",
    "attention_backend", "kv_dtype", "block_size",
)


def attention_case_params_to_signature(
    params: dict[str, Any],
    *,
    framework: str,
    framework_version: str,
    kernel_source: str,
    attention_backend: str | None = None,
    kv_dtype: str | None = None,
    block_size: int | None = None,
) -> OperatorSignature:
    """collector attention Case.params + RawRecord top-level → OperatorSignature."""
    phase = params["phase"]
    if phase == "prefill":
        op_subtype = "prefill"
        bs = int(params["batch_size"])
        isl = int(params["isl"])
        shape_fields = {
            "num_tokens": bs * isl,
            "num_seqs": bs,
            "q_len": isl,
            "kv_len": isl,
            "num_q_heads": params["num_heads"],
            "num_kv_heads": params["num_kv_heads"],
            "head_dim": params["head_dim"],
        }
    elif phase == "decode":
        op_subtype = "decode"
        n = int(params["n_decode"])
        shape_fields = {
            "num_tokens": n,
            "num_seqs": n,
            "q_len": 1,
            "kv_len": int(params["kv_decode"]),
            "num_q_heads": params["num_heads"],
            "num_kv_heads": params["num_kv_heads"],
            "head_dim": params["head_dim"],
        }
    else:
        raise NotImplementedError(
            f"attention canonicalizer stage 2 仅 prefill / decode, got {phase!r}"
        )

    runtime_fields = {
        "framework": framework,
        "framework_version": framework_version,
        "execution_mode": params["execution_mode"],
        "kernel_source": kernel_source,
        "attention_backend": attention_backend,
        "kv_dtype": kv_dtype,
        "block_size": block_size,
    }
    return OperatorSignature(
        op_kind="attention",
        op_subtype=op_subtype,
        dtype=params["dtype"],
        shape=to_canonical(shape_fields),
        parallel=to_canonical({"tp": int(params["tp"])}),
        runtime=to_canonical(runtime_fields),
    )


def attention_operator_to_signature(op: Any) -> OperatorSignature:
    if op.op_kind != "attention":
        raise ValueError(f"expected op_kind=attention, got {op.op_kind!r}")
    return OperatorSignature(
        op_kind="attention",
        op_subtype=op.op_subtype,
        dtype=op.dtype,
        shape=to_canonical(project(op.shape, _SHAPE_KEYS)),
        parallel=to_canonical(project(op.parallel, _PARALLEL_KEYS)),
        runtime=to_canonical(project(op.runtime, _RUNTIME_KEYS)),
    )
