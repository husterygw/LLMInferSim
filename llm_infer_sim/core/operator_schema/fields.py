"""Phase 1 signature-field registry + validating builder (op_plan §2).

The DB-miss root-cause (2026-05) was silent signature-field mismatch: a query op
emitting ``message_bytes`` typo'd, or a stray key, just dropped out via ``project``
and missed every record with no error. This registry makes the field names a
checked contract: building a signature with an unregistered field raises, so a
typo fails loudly in tests instead of becoming a silent roofline fallback.

Field sets are sourced from each op_kind's canonicalizer (single source of truth
— do not duplicate the key tuples here, or they drift).
"""
from __future__ import annotations

from typing import Any, Mapping

from llm_infer_sim.core.operator_schema.attention import (
    _PARALLEL_KEYS as _ATTN_PARALLEL,
    _RUNTIME_KEYS as _ATTN_RUNTIME,
    _SHAPE_KEYS as _ATTN_SHAPE,
)
from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.collective import (
    _PARALLEL_KEYS as _COLL_PARALLEL,
    _RUNTIME_KEYS as _COLL_RUNTIME,
    _SHAPE_KEYS as _COLL_SHAPE,
)
from llm_infer_sim.core.operator_schema.gemm import (
    _PARALLEL_KEYS as _GEMM_PARALLEL,
    _RUNTIME_KEYS as _GEMM_RUNTIME,
    _SHAPE_KEYS as _GEMM_SHAPE,
)
from llm_infer_sim.core.operator_schema.moe import (
    _PARALLEL_KEYS as _MOE_PARALLEL,
    _RUNTIME_KEYS as _MOE_RUNTIME,
    _SHAPE_KEYS as _MOE_SHAPE,
)
from llm_infer_sim.core.operator_schema.signature import OperatorSignature


class SignatureFieldError(ValueError):
    """A signature was built with a field name not registered for its op_kind."""


#: op_kind -> {"shape"/"parallel"/"runtime" -> frozenset of allowed field names}.
SIGNATURE_FIELDS: dict[str, dict[str, frozenset[str]]] = {
    "gemm": {
        "shape": frozenset(_GEMM_SHAPE),
        "parallel": frozenset(_GEMM_PARALLEL),
        "runtime": frozenset(_GEMM_RUNTIME),
    },
    "attention": {
        "shape": frozenset(_ATTN_SHAPE),
        "parallel": frozenset(_ATTN_PARALLEL),
        "runtime": frozenset(_ATTN_RUNTIME),
    },
    "moe": {
        "shape": frozenset(_MOE_SHAPE),
        "parallel": frozenset(_MOE_PARALLEL),
        "runtime": frozenset(_MOE_RUNTIME),
    },
    "collective": {
        "shape": frozenset(_COLL_SHAPE),
        "parallel": frozenset(_COLL_PARALLEL),
        "runtime": frozenset(_COLL_RUNTIME),
    },
}


def _validate(op_kind: str, part: str, fields: Mapping[str, Any], allowed: frozenset[str]) -> None:
    # None means "field not applicable" (canonicalizer convention) — ignore.
    present = {k for k, v in fields.items() if v is not None}
    unknown = present - allowed
    if unknown:
        raise SignatureFieldError(
            f"{op_kind}.{part}: unregistered signature field(s) {sorted(unknown)} "
            f"(registered: {sorted(allowed)}). A typo here would silently miss the "
            f"DB — add the field to its canonicalizer or fix the name."
        )


def build_validated_signature(
    *,
    op_kind: str,
    op_subtype: str,
    dtype: str,
    shape: Mapping[str, Any],
    parallel: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> OperatorSignature:
    """Build an OperatorSignature, rejecting unregistered field names.

    Mirrors the existing canonicalizers' projection/canonicalization, but adds
    the registry check. Intended as the single signature-construction entry once
    the per-op canonicalizers are migrated onto it (Phase 2-4).
    """
    if op_kind not in SIGNATURE_FIELDS:
        raise SignatureFieldError(
            f"op_kind {op_kind!r} not registered (known: {sorted(SIGNATURE_FIELDS)})."
        )
    allowed = SIGNATURE_FIELDS[op_kind]
    _validate(op_kind, "shape", shape, allowed["shape"])
    _validate(op_kind, "parallel", parallel, allowed["parallel"])
    _validate(op_kind, "runtime", runtime, allowed["runtime"])
    return OperatorSignature(
        op_kind=op_kind,
        op_subtype=op_subtype,
        dtype=dtype,
        shape=to_canonical(project(dict(shape), tuple(sorted(allowed["shape"])))),
        parallel=to_canonical(project(dict(parallel), tuple(sorted(allowed["parallel"])))),
        runtime=to_canonical(project(dict(runtime), tuple(sorted(allowed["runtime"])))),
    )
