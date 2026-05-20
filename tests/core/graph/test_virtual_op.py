"""V3 §4.3 VirtualOp 单测 — 阶段 1 Step 1.1.

锁住:
  - 字段集 (必填 + 默认)
  - frozen 行为
  - dict 字段可携带任意 shape / parallel / runtime / formula key
"""
from __future__ import annotations

import dataclasses

import pytest

from llm_infer_sim.core.graph.virtual_op import VirtualOp


def _make_gemm_op(**overrides) -> VirtualOp:
    defaults = dict(
        name="qkv_proj",
        op_kind="gemm",
        op_subtype="qkv_proj",
        phase="prefill",
        layer_idx=0,
        dtype="bf16",
        shape={"m": 128, "n": 6144, "k": 2560},
        parallel={"tp": 1},
        runtime={
            "framework": "vllm",
            "framework_version": "0.20.1",
            "execution_mode": "eager",
            "kernel_source": "vllm_default",
        },
        formula={
            "flops": 2 * 128 * 6144 * 2560,
            "load_weight": 6144 * 2560 * 2,
            "load_act": 128 * 2560 * 2,
            "store_act": 128 * 6144 * 2,
            "op_precision": "bf16",
            "op_category": "matmul",
        },
    )
    defaults.update(overrides)
    return VirtualOp(**defaults)


def test_required_fields_present():
    op = _make_gemm_op()
    assert op.name == "qkv_proj"
    assert op.op_kind == "gemm"
    assert op.op_subtype == "qkv_proj"
    assert op.phase == "prefill"
    assert op.layer_idx == 0
    assert op.dtype == "bf16"
    assert op.shape["m"] == 128
    assert op.parallel["tp"] == 1
    assert op.runtime["execution_mode"] == "eager"
    assert op.formula["op_category"] == "matmul"


def test_optional_defaults():
    op = _make_gemm_op()
    assert op.dependencies == ()
    assert op.tags == ()


def test_frozen():
    op = _make_gemm_op()
    with pytest.raises(dataclasses.FrozenInstanceError):
        op.name = "other"  # type: ignore[misc]


def test_dependencies_and_tags_are_tuples():
    op = _make_gemm_op(
        dependencies=("attn_norm",),
        tags=("merged_qkv", "stage_1"),
    )
    assert op.dependencies == ("attn_norm",)
    assert op.tags == ("merged_qkv", "stage_1")


def test_attention_shape_keys():
    """V3 §5.3 attention shape: num_tokens / num_seqs / q_len / kv_len / heads / head_dim."""
    op = VirtualOp(
        name="attn_prefill",
        op_kind="attention",
        op_subtype="prefill",
        phase="prefill",
        layer_idx=0,
        dtype="bf16",
        shape={
            "num_tokens": 128, "num_seqs": 1,
            "q_len": 128, "kv_len": 128,
            "num_q_heads": 32, "num_kv_heads": 8,
            "head_dim": 128,
        },
        parallel={"tp": 1},
        runtime={
            "framework": "vllm", "framework_version": "0.20.1",
            "execution_mode": "eager", "kernel_source": "vllm_default",
            "attention_backend": "flash_attn", "kv_dtype": "bf16",
            "block_size": 16,
        },
        formula={
            "flops": 12345, "load_act": 1, "store_act": 1,
            "load_kv_cache": 1, "store_kv_cache": 1,
            "op_precision": "bf16", "op_category": "attention",
        },
    )
    assert op.shape["num_q_heads"] == 32
    assert op.runtime["attention_backend"] == "flash_attn"


def test_collective_op_carries_comm_bytes():
    op = VirtualOp(
        name="attn_allreduce",
        op_kind="collective",
        op_subtype="allreduce",
        phase="prefill",
        layer_idx=0,
        dtype="bf16",
        shape={"message_bytes": 128 * 2560 * 2},
        parallel={"world_size": 2, "tp": 2},
        runtime={"backend": "nccl", "topology": "single_node"},
        formula={
            "comm_bytes": 128 * 2560 * 2,
            "comm_type": "allreduce",
            "op_category": "communication",
            "op_precision": "bf16",
        },
    )
    assert op.op_kind == "collective"
    assert op.formula["comm_type"] == "allreduce"
