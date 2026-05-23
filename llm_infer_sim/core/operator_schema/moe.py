"""MoE canonicalizer — V3 §5.2 / IMPL_PLAN §2.3.

字段:
    op_kind   = moe
    op_subtype= fused_moe
    dtype
    shape     = {num_tokens, hidden, moe_intermediate, topk, num_experts,
                 routing_distribution, power_law_alpha}
    parallel  = {tp, ep}
    runtime   = {framework, framework_version, execution_mode, kernel_source}

routing_distribution 必须进入 key — balanced 与 power_law 不能互相命中.
"""
from __future__ import annotations

from typing import Any

from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature

_SHAPE_KEYS = (
    "num_tokens", "hidden", "moe_intermediate", "topk", "num_experts",
    "routing_distribution", "power_law_alpha",
)
_PARALLEL_KEYS = ("tp", "ep")
_RUNTIME_KEYS = ("framework", "framework_version", "execution_mode", "kernel_source")


def moe_case_params_to_signature(
    params: dict[str, Any],
    *,
    framework: str,
    framework_version: str,
    kernel_source: str,
) -> OperatorSignature:
    """collector MoE Case.params + RawRecord top-level → OperatorSignature.

    Case.params 必含: num_tokens, hidden, moe_intermediate, topk, num_experts,
                      tp, ep, routing_distribution, power_law_alpha, dtype, execution_mode
    """
    runtime = {
        "framework": framework,
        "framework_version": framework_version,
        "execution_mode": params["execution_mode"],
        "kernel_source": kernel_source,
    }
    return OperatorSignature(
        op_kind="moe",
        op_subtype="fused_moe",
        dtype=params["dtype"],
        shape=to_canonical(project(params, _SHAPE_KEYS)),
        parallel=to_canonical(project(params, _PARALLEL_KEYS)),
        runtime=to_canonical(runtime),
    )


def moe_virtual_op_to_signature(op: Any) -> OperatorSignature:
    """runtime operator descriptor → OperatorSignature.

    qwen.py / deepseek.py 直接构造 FusedMoE (op_kind=moe), 此 canonicalizer
    把 FusedMoE 转成 OperatorDB signature.
    """
    if op.op_kind != "moe":
        raise ValueError(f"expected op_kind=moe, got {op.op_kind!r}")
    return OperatorSignature(
        op_kind="moe",
        op_subtype=op.op_subtype,
        dtype=op.dtype,
        shape=to_canonical(project(op.shape, _SHAPE_KEYS)),
        parallel=to_canonical(project(op.parallel, _PARALLEL_KEYS)),
        runtime=to_canonical(project(op.runtime, _RUNTIME_KEYS)),
    )
