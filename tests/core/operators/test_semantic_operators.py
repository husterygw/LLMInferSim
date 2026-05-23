from __future__ import annotations

import pytest

from llm_infer_sim.core.operator_schema.attention import attention_case_params_to_signature
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.operators import (
    Attention,
    Collective,
    ElementWise,
    Embedding,
    RooflineOperator,
    FusedMoE,
    KVTransfer,
    Norm,
)
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig


def _formula() -> RooflineSpec:
    return RooflineSpec(op_category="activation", flops=1, load_act=2, store_act=2)


def _ctx(tp=1, ep=1, mode="eager", framework_version="0.20.1"):
    return build_operator_context(
        ModelConfig(),
        DeployConfig(
            tp_size=tp, ep_size=ep, execution_mode=mode,
            backend="vllm", backend_version=framework_version,
        ),
        get_hardware_profile("RTX_4090"),
    )


def test_non_db_ops_raise_for_signature():
    """Norm / ElementWise / Embedding / KVTransfer 不进 OperatorDB; signature() 应抛."""
    ctx = _ctx()
    norm = Norm(
        name="attn_norm", op_subtype="attn_norm",
        phase="prefill", layer_idx=0,
        tokens=1, hidden=128, ctx=ctx,
    )
    with pytest.raises(ValueError, match="signature contract"):
        norm.signature()

    ew = ElementWise(
        name="attn_add", op_subtype="attn_add",
        phase="prefill", layer_idx=0,
        tokens=1, hidden=128, ctx=ctx,
    )
    with pytest.raises(ValueError, match="signature contract"):
        ew.signature()

    embed = Embedding(
        name="embedding", phase="prefill", layer_idx=None,
        tokens=1, vocab_size=32000, hidden=128, ctx=ctx,
    )
    with pytest.raises(ValueError, match="signature contract"):
        embed.signature()

    # KVTransfer 走 legacy RooflineOperator
    kv = KVTransfer(
        name="kv_transfer", op_kind="kv_transfer", op_subtype="kv_transfer",
        phase="prefill", layer_idx=0, dtype="bf16",
        shape_fields={"tokens": 1}, parallel_fields={"tp": 1},
        runtime_fields={"framework": "vllm"},
        roofline_spec_value=_formula(),
    )
    with pytest.raises(ValueError, match="signature contract"):
        kv.signature()


def test_attention_op_signature_matches_collector_contract():
    op = Attention(
        name="attention", op_subtype="prefill",
        phase="prefill", layer_idx=0,
        num_tokens=128, num_seqs=1,
        q_len=128, kv_len=128,
        num_q_heads=32, num_kv_heads=8, head_dim=128,
        attention_backend="flash_attn", kv_dtype="bf16", block_size=16,
        kernel_source="vllm_flash_attn",
        ctx=_ctx(),
        roofline_spec_value=RooflineSpec(op_category="attention", flops=1),
    )
    expected = attention_case_params_to_signature(
        {
            "phase": "prefill",
            "batch_size": 1,
            "isl": 128,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "dtype": "bf16",
            "tp": 1,
            "execution_mode": "eager",
        },
        framework="vllm",
        framework_version="0.20.1",
        kernel_source="vllm_flash_attn",
        attention_backend="flash_attn",
        kv_dtype="bf16",
        block_size=16,
    )
    assert op.signature() == expected


def test_collective_and_moe_ops_have_db_signatures():
    collective = Collective(
        name="allreduce", op_subtype="allreduce",
        phase="decode", layer_idx=0,
        message_bytes=1024, world_size=2,
        ctx=_ctx(tp=2),
        comm_backend="nccl", topology="single_node",
        kernel_source="nccl",
        roofline_spec_value=RooflineSpec(op_category="communication", comm_bytes=1024),
    )
    assert collective.signature().op_kind == "collective"

    # FusedMoE 在 #158 Step 3 后变 ctx-based, 直接用 shape 字段构造
    moe = FusedMoE(
        name="fused_moe", op_subtype="fused_moe",
        phase="decode", layer_idx=0,
        num_tokens=16, hidden=2048,
        moe_intermediate=768, topk=8,
        num_experts=128,
        routing_distribution="balanced",
        power_law_alpha=1.0,
        ctx=_ctx(tp=1, ep=8, mode="cudagraph"),
        roofline_spec_value=RooflineSpec(op_category="matmul", flops=1),
    )
    assert moe.signature().op_kind == "moe"
