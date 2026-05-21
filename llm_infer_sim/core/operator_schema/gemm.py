"""GEMM canonicalizer — V3 §5.1 / IMPL_PLAN §2.2.

字段:
    op_kind   = gemm
    op_subtype= qkv_proj / o_proj / gate_up_proj / down_proj / lm_head / router / ...
    dtype
    shape     = {m, n, k}
    parallel  = {tp}
    runtime   = {framework, framework_version, execution_mode, kernel_source}
"""
from __future__ import annotations

from typing import Any

from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature

_SHAPE_KEYS = ("m", "n", "k")
_PARALLEL_KEYS = ("tp",)
_RUNTIME_KEYS = ("framework", "framework_version", "execution_mode", "kernel_source")


def gemm_case_params_to_signature(
    params: dict[str, Any],
    *,
    framework: str,
    framework_version: str,
    kernel_source: str,
) -> OperatorSignature:
    """collector Case.params + RawRecord top-level → OperatorSignature.

    Case.params 必含: op_subtype, m, n, k, dtype, tp, execution_mode
    RawRecord 顶层提供: framework, framework_version, kernel_source
    """
    runtime = {
        "framework": framework,
        "framework_version": framework_version,
        "execution_mode": params["execution_mode"],
        "kernel_source": kernel_source,
    }
    return OperatorSignature(
        op_kind="gemm",
        op_subtype=params["op_subtype"],
        dtype=params["dtype"],
        shape=to_canonical(project(params, _SHAPE_KEYS)),
        parallel=to_canonical(project(params, _PARALLEL_KEYS)),
        runtime=to_canonical(runtime),
    )


def gemm_virtual_op_to_signature(op: Any) -> OperatorSignature:
    """runtime operator descriptor → OperatorSignature.

    op.shape 必含 m/n/k. op.parallel 必含 tp. op.runtime 必含 framework/version/mode/kernel_source.
    """
    if op.op_kind != "gemm":
        raise ValueError(f"expected op_kind=gemm, got {op.op_kind!r}")
    return OperatorSignature(
        op_kind="gemm",
        op_subtype=op.op_subtype,
        dtype=op.dtype,
        shape=to_canonical(project(op.shape, _SHAPE_KEYS)),
        parallel=to_canonical(project(op.parallel, _PARALLEL_KEYS)),
        runtime=to_canonical(project(op.runtime, _RUNTIME_KEYS)),
    )
