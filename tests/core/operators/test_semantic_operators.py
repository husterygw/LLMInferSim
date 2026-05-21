from __future__ import annotations

import pytest

from llm_infer_sim.core.operator_schema.attention import attention_case_params_to_signature
from llm_infer_sim.core.operators.ops import (
    AttentionOp,
    CollectiveOp,
    ElementwiseOp,
    EmbeddingOp,
    FusedMoeOp,
    KvTransferOp,
    NormOp,
)
from llm_infer_sim.core.operators.specs import OperatorFormula


def _formula() -> OperatorFormula:
    return OperatorFormula(op_category="activation", flops=1, load_act=2, store_act=2)


def test_non_db_ops_raise_for_signature():
    for cls, kind in (
        (EmbeddingOp, "embedding"),
        (NormOp, "norm"),
        (ElementwiseOp, "elementwise"),
        (KvTransferOp, "kv_transfer"),
    ):
        op = cls(
            name=kind,
            op_kind=kind,
            op_subtype=kind,
            phase="prefill",
            layer_idx=None,
            dtype="bf16",
            shape_fields={"tokens": 1},
            parallel_fields={"tp": 1},
            runtime_fields={"framework": "vllm"},
            formula_value=_formula(),
        )
        assert op.shape["tokens"] == 1
        assert op.formula().flops == 1
        with pytest.raises(ValueError, match="OperatorDB signature contract"):
            op.signature()


def test_attention_op_signature_matches_collector_contract():
    op = AttentionOp(
        name="layer0_attention",
        op_kind="attention",
        op_subtype="prefill",
        phase="prefill",
        layer_idx=0,
        dtype="bf16",
        shape_fields={
            "num_tokens": 128,
            "num_seqs": 1,
            "q_len": 128,
            "kv_len": 128,
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        parallel_fields={"tp": 1},
        runtime_fields={
            "framework": "vllm",
            "framework_version": "0.20.1",
            "execution_mode": "eager",
            "kernel_source": "vllm_flash_attn",
            "attention_backend": "flash_attn",
            "kv_dtype": "bf16",
            "block_size": 16,
        },
        formula_value=OperatorFormula(op_category="attention", flops=1),
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
    collective = CollectiveOp(
        name="allreduce",
        op_kind="collective",
        op_subtype="allreduce",
        phase="decode",
        layer_idx=0,
        dtype="bf16",
        shape_fields={"message_bytes": 1024},
        parallel_fields={
            "world_size": 2,
            "tp": 2,
            "ep": None,
            "node_count": 1,
            "gpus_per_node": 2,
        },
        runtime_fields={
            "framework": "vllm",
            "framework_version": "0.20.1",
            "backend": "nccl",
            "algo": None,
            "protocol": None,
            "topology": "single_node",
            "execution_mode": "eager",
            "kernel_source": "nccl",
        },
        formula_value=OperatorFormula(op_category="communication", comm_bytes=1024),
    )
    assert collective.signature().op_kind == "collective"

    moe = FusedMoeOp(
        name="fused_moe",
        op_kind="moe",
        op_subtype="fused_moe",
        phase="decode",
        layer_idx=0,
        dtype="bf16",
        shape_fields={
            "num_tokens": 16,
            "hidden": 2048,
            "moe_intermediate": 768,
            "topk": 8,
            "num_experts": 128,
            "routing_distribution": "balanced",
            "power_law_alpha": 1.0,
        },
        parallel_fields={"tp": 1, "ep": 8},
        runtime_fields={
            "framework": "vllm",
            "framework_version": "0.20.1",
            "execution_mode": "cudagraph",
            "kernel_source": "vllm_fused_moe",
        },
        formula_value=OperatorFormula(op_category="matmul", flops=1),
    )
    assert moe.signature().op_kind == "moe"
