"""MoE canonicalizer 单测 — Step 2.3.

锁住:
  - balanced 与 power_law signature 不同 (IMPL_PLAN §2.4 测试点 2)
  - power_law alpha 不同 → 不同 signature
  - tp / ep 都进 parallel key
  - collector ↔ runtime signature 一致 (Stage 4 后才会真正生成 runtime MoE op)
"""
from __future__ import annotations

from llm_infer_sim.core.operator_schema.moe import (
    moe_case_params_to_signature,
    moe_virtual_op_to_signature,
)
from llm_infer_sim.core.operators.ops import FusedMoeOp
from llm_infer_sim.core.operators.specs import OperatorFormula


_CTX = dict(framework="vllm", framework_version="0.20.1", kernel_source="vllm_fused_moe")


def _moe_case(routing="balanced", alpha=0.0, tp=1, ep=4, mode="eager"):
    return {
        "num_tokens": 128, "hidden": 2048,
        "moe_intermediate": 768, "topk": 8, "num_experts": 128,
        "tp": tp, "ep": ep,
        "routing_distribution": routing,
        "power_law_alpha": alpha,
        "dtype": "bf16", "execution_mode": mode,
    }


def _moe_op(routing="balanced", alpha=0.0, tp=1, ep=4, mode="eager"):
    return FusedMoeOp(
        name="layer0_fused_moe", op_kind="moe", op_subtype="fused_moe",
        phase="prefill", layer_idx=0, dtype="bf16",
        shape_fields={
            "num_tokens": 128, "hidden": 2048,
            "moe_intermediate": 768, "topk": 8, "num_experts": 128,
            "routing_distribution": routing,
            "power_law_alpha": alpha,
        },
        parallel_fields={"tp": tp, "ep": ep},
        runtime_fields={
            "framework": "vllm", "framework_version": "0.20.1",
            "execution_mode": mode, "kernel_source": "vllm_fused_moe",
        },
        formula_value=OperatorFormula(flops=1, op_category="matmul"),
    )


def test_collector_and_runtime_signature_match():
    sig_c = moe_case_params_to_signature(_moe_case(), **_CTX)
    sig_r = moe_virtual_op_to_signature(_moe_op())
    assert sig_c == sig_r
    assert sig_c.stable_hash() == sig_r.stable_hash()


def test_balanced_vs_power_law_differ():
    sig_b = moe_case_params_to_signature(
        _moe_case(routing="balanced", alpha=0.0), **_CTX,
    )
    sig_p = moe_case_params_to_signature(
        _moe_case(routing="power_law", alpha=1.2), **_CTX,
    )
    assert sig_b != sig_p


def test_power_law_alpha_difference_changes_signature():
    """power_law alpha=1.01 vs 1.2 应该是不同 signature."""
    sig_a = moe_case_params_to_signature(
        _moe_case(routing="power_law", alpha=1.01), **_CTX,
    )
    sig_b = moe_case_params_to_signature(
        _moe_case(routing="power_law", alpha=1.2), **_CTX,
    )
    assert sig_a != sig_b


def test_tp_and_ep_in_parallel_key():
    sig_a = moe_case_params_to_signature(_moe_case(tp=1, ep=4), **_CTX)
    sig_b = moe_case_params_to_signature(_moe_case(tp=4, ep=1), **_CTX)
    assert sig_a != sig_b


def test_eager_cudagraph_differ():
    sig_e = moe_case_params_to_signature(_moe_case(mode="eager"), **_CTX)
    sig_g = moe_case_params_to_signature(_moe_case(mode="cudagraph"), **_CTX)
    assert sig_e != sig_g
