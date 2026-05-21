"""Operator Schema Contract — V3 §5 / IMPL_PLAN §2.

collector Case.params / runtime Operator / OperatorDB Query 必须 canonicalize 到
同一个 OperatorSignature.
"""
from llm_infer_sim.core.operator_schema.attention import (
    attention_case_params_to_signature,
    attention_virtual_op_to_signature,
)
from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.collective import (
    collective_case_params_to_signature,
    collective_virtual_op_to_signature,
)
from llm_infer_sim.core.operator_schema.gemm import (
    gemm_case_params_to_signature,
    gemm_virtual_op_to_signature,
)
from llm_infer_sim.core.operator_schema.moe import (
    moe_case_params_to_signature,
    moe_virtual_op_to_signature,
)
from llm_infer_sim.core.operator_schema.signature import OperatorSignature


def operator_to_signature(op) -> OperatorSignature:
    """Operator or legacy descriptor → OperatorSignature.

    OperatorDB 查询只针对 GEMM / Attention / MoE / Collective (V3 §5 列出的四类);
    其他 (norm / elementwise / embedding) 走 RooflineBackend 不需要 signature.
    """
    signature_fn = getattr(op, "signature", None)
    if callable(signature_fn):
        return signature_fn()

    dispatch = {
        "gemm": gemm_virtual_op_to_signature,
        "attention": attention_virtual_op_to_signature,
        "moe": moe_virtual_op_to_signature,
        "collective": collective_virtual_op_to_signature,
    }
    fn = dispatch.get(op.op_kind)
    if fn is None:
        raise ValueError(
            f"op_kind {op.op_kind!r} not in OperatorDB signature contract; "
            f"only gemm/attention/moe/collective enter DB (V3 §5)."
        )
    return fn(op)


def virtual_op_to_signature(op) -> OperatorSignature:
    """Compatibility wrapper for legacy call sites."""
    return operator_to_signature(op)


__all__ = [
    "OperatorSignature",
    "to_canonical",
    "project",
    "operator_to_signature",
    "virtual_op_to_signature",
    "gemm_case_params_to_signature",
    "gemm_virtual_op_to_signature",
    "attention_case_params_to_signature",
    "attention_virtual_op_to_signature",
    "moe_case_params_to_signature",
    "moe_virtual_op_to_signature",
    "collective_case_params_to_signature",
    "collective_virtual_op_to_signature",
]
