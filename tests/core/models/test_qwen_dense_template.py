"""QwenModelGraphTemplate 单测 — IMPL_PLAN §1.5.

锁住:
  - Qwen3-4B prefill / decode 生成 op list
  - per-layer 顺序 + 跨 layer 数量
  - 各 op 携带 op_kind / op_subtype / shape / parallel / runtime
  - search-ready: TP 改变后 shape/parallel 跟变, TP>1 生成 dense allreduce
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.models.qwen import QwenModelGraphTemplate
from llm_infer_sim.core.operators import Collective, GEMM
from llm_infer_sim.core.operators.base import Operator
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload,
    RequestWorkload,
    StepPhase,
)


def _qwen3_4b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-4B",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def _make_ctx(model: ModelConfig, deploy: DeployConfig):
    from llm_infer_sim.core.operators.context import build_operator_context
    hw = get_hardware_profile("RTX_4090")
    return build_operator_context(model, deploy, hw)


def _prefill_step(isl: int = 128) -> StepShape:
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=isl, context_len=0,
        )],
        num_prefill_tokens=isl, num_decode_tokens=0,
        total_scheduled_tokens=isl,
        num_prefill_requests=1, num_decode_requests=0,
    )
    return StepShape.from_workload(wl, DeployConfig())


def _decode_step(n: int = 8, ctx: int = 1024) -> StepShape:
    wl = GlobalStepWorkload(
        step_id=1, phase=StepPhase.DECODE,
        requests=[
            RequestWorkload(
                request_id=f"d{i}", phase=StepPhase.DECODE,
                num_tokens=1, context_len=ctx,
            )
            for i in range(n)
        ],
        num_prefill_tokens=0, num_decode_tokens=n,
        total_scheduled_tokens=n,
        num_prefill_requests=0, num_decode_requests=n,
    )
    return StepShape.from_workload(wl, DeployConfig())


PER_LAYER_SUBTYPES = (
    "attn_norm", "qkv_proj", "rope", "prefill",  # attention prefill
    "o_proj", "attn_add", "mlp_norm", "gate_up_proj",
    "mlp_act", "down_proj", "mlp_add",
)


def test_prefill_op_count():
    """1 embedding + num_layers × 11 per-layer + 1 lm_head."""
    model = _qwen3_4b()
    deploy = DeployConfig()
    ctx = _make_ctx(model, deploy)
    step = _prefill_step()
    plan = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(step)
    expected = 1 + model.num_layers * 11 + 1
    assert len(plan.ops) == expected


def test_decode_op_count():
    model = _qwen3_4b()
    deploy = DeployConfig()
    ctx = _make_ctx(model, deploy)
    step = _decode_step()
    plan = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(step)
    expected = 1 + model.num_layers * 11 + 1
    assert len(plan.ops) == expected


def test_per_layer_order_first_layer():
    """每层 11 个 op 顺序应该是: attn_norm/qkv/rope/attn/o/attn_add/mlp_norm/gu/act/down/mlp_add.

    Grouped trace: per-layer ops 从 groups 中 filter layer_idx ∈ g.layer_indices 的代表 op,
    顺序就是 build_grouped_step 的 group 顺序.
    """
    model = _qwen3_4b()
    deploy = DeployConfig()
    ctx = _make_ctx(model, deploy)
    plan = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(_prefill_step())
    layer0 = [g.op for g in plan.groups if 0 in g.layer_indices]
    expected_subtypes = [
        "attn_norm", "qkv_proj", "rope", "prefill",
        "o_proj", "attn_add", "mlp_norm", "gate_up_proj",
        "mlp_act", "down_proj", "mlp_add",
    ]
    assert [op.op_subtype for op in layer0] == expected_subtypes
    # 代表 op 的 layer_idx 是 0 (build_grouped_step 用 rep=layer_indices[0])
    for op in layer0:
        assert op.layer_idx == 0


def test_embedding_and_lm_head_are_at_boundaries():
    model = _qwen3_4b()
    deploy = DeployConfig()
    ctx = _make_ctx(model, deploy)
    plan = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(_prefill_step())
    assert plan.ops[0].op_kind == "embedding"
    assert plan.ops[0].op_subtype == "embedding"
    assert plan.ops[-1].op_kind == "gemm"
    assert plan.ops[-1].op_subtype == "lm_head"


def test_qkv_proj_shape_matches_gqa():
    """Qwen3-4B: num_heads=32, num_kv_heads=8, head_dim=128 → QKV n = (32 + 16) × 128 = 6144."""
    model = _qwen3_4b()
    deploy = DeployConfig()
    ctx = _make_ctx(model, deploy)
    plan = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(_prefill_step(isl=128))
    qkv_ops = [op for op in plan.ops if op.op_subtype == "qkv_proj"]
    assert qkv_ops, "no qkv_proj op generated"
    op = qkv_ops[0]
    assert isinstance(op, GEMM)
    assert op.shape["m"] == 128
    assert op.shape["k"] == model.hidden_dim
    # Q dim = 32 × 128 = 4096, K = V = 8 × 128 = 1024; total n = 4096 + 2*1024 = 6144
    assert op.shape["n"] == 32 * 128 + 2 * 8 * 128
    assert op.parallel["tp"] == 1
    assert op.runtime["execution_mode"] == "eager"


def test_attention_op_carries_full_shape():
    """V3 §5.3: attention shape 必带 num_tokens/num_seqs/q_len/kv_len/heads/head_dim."""
    model = _qwen3_4b()
    deploy = DeployConfig()
    ctx = _make_ctx(model, deploy)
    plan = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(_prefill_step(isl=2048))
    attn_ops = [op for op in plan.ops if op.op_kind == "attention"]
    assert len(attn_ops) == model.num_layers
    op = attn_ops[0]
    assert op.op_subtype == "prefill"
    assert op.shape["q_len"] == 2048
    assert op.shape["kv_len"] == 2048
    assert op.shape["num_q_heads"] == 32
    assert op.shape["num_kv_heads"] == 8
    assert op.shape["head_dim"] == 128
    assert op.runtime["attention_backend"] == "flash_attn"
    assert op.runtime["block_size"] == 16


def test_decode_attention_subtype_and_kv_len():
    model = _qwen3_4b()
    deploy = DeployConfig()
    ctx = _make_ctx(model, deploy)
    plan = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(_decode_step(n=8, ctx=1024))
    attn = [op for op in plan.ops if op.op_kind == "attention"][0]
    assert attn.op_subtype == "decode"
    assert attn.shape["q_len"] == 1
    assert attn.shape["kv_len"] == 1024
    assert attn.shape["num_seqs"] == 8


def test_lm_head_tokens_equals_num_requests():
    """prefill: 每 req 1 个采样 token; decode: 同."""
    model = _qwen3_4b()
    deploy = DeployConfig()
    ctx = _make_ctx(model, deploy)
    # prefill bs=1
    plan_p = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(_prefill_step(isl=2048))
    head_p = [op for op in plan_p.ops if op.op_subtype == "lm_head"][0]
    assert head_p.shape["m"] == 1
    # decode bs=8
    plan_d = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(_decode_step(n=8))
    head_d = [op for op in plan_d.ops if op.op_subtype == "lm_head"][0]
    assert head_d.shape["m"] == 8


def test_tp_affects_qkv_shape():
    """search-ready: TP 改变后 GEMM n 维按 tp 切."""
    model = _qwen3_4b()
    step = _prefill_step(isl=128)

    plan_tp1 = QwenModelGraphTemplate(
        model, ctx=_make_ctx(model, DeployConfig(tp_size=1)),
    ).build_grouped_step(step)
    # deploy 改 tp=2 后 step.execution_mode 不变, step_shape 可复用
    plan_tp2 = QwenModelGraphTemplate(
        model, ctx=_make_ctx(model, DeployConfig(tp_size=2)),
    ).build_grouped_step(step)

    qkv1 = [op for op in plan_tp1.ops if op.op_subtype == "qkv_proj"][0]
    qkv2 = [op for op in plan_tp2.ops if op.op_subtype == "qkv_proj"][0]
    # tp=2 后 num_q_heads_per_tp=16, num_kv_heads_per_tp=4 → n = (16+8)*128 = 3072 = qkv1.n/2
    assert qkv2.shape["n"] == qkv1.shape["n"] // 2
    assert qkv1.parallel["tp"] == 1
    assert qkv2.parallel["tp"] == 2


def test_tp1_has_no_dense_allreduce():
    model = _qwen3_4b()
    step = _prefill_step(isl=128)
    plan = QwenModelGraphTemplate(
        model, ctx=_make_ctx(model, DeployConfig(tp_size=1)),
    ).build_grouped_step(step)
    names = {op.name for op in plan.ops}
    assert "tp_o_proj_allreduce" not in names
    assert "tp_down_proj_allreduce" not in names


def test_tp_dense_allreduce_grouped_shape_and_bytes():
    model = _qwen3_4b()
    deploy = DeployConfig(tp_size=2)
    ctx = _make_ctx(model, deploy)
    step = _prefill_step(isl=128)
    plan = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(step)

    groups = {g.op.name: g for g in plan.groups}
    expected_bytes = int(step.total_tokens * model.hidden_dim * ctx.a_byte)
    for name in ("tp_o_proj_allreduce", "tp_down_proj_allreduce"):
        group = groups[name]
        op = group.op
        assert isinstance(op, Collective)
        assert op.op_subtype == "allreduce"
        assert op.message_bytes == expected_bytes
        assert op.parallel["world_size"] == 2
        assert group.count == model.num_layers
        assert group.layer_indices == tuple(range(model.num_layers))


def test_all_ops_have_required_metadata():
    """所有 runtime op 必含 op_kind/op_subtype/shape/parallel/runtime/formula."""
    model = _qwen3_4b()
    deploy = DeployConfig()
    ctx = _make_ctx(model, deploy)
    plan = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(_prefill_step())
    for op in plan.ops:
        assert isinstance(op, Operator)
        assert op.op_kind and op.op_subtype
        assert op.shape, f"{op.name} no shape"
        assert op.parallel, f"{op.name} no parallel"
        assert op.runtime, f"{op.name} no runtime"
        assert op.roofline_spec, f"{op.name} no roofline_spec"


def test_plan_metadata_carries_model_info():
    model = _qwen3_4b()
    deploy = DeployConfig(execution_mode="cudagraph")
    ctx = _make_ctx(model, deploy)
    step = StepShape.from_workload(
        GlobalStepWorkload(
            step_id=42, phase=StepPhase.PREFILL,
            requests=[RequestWorkload(
                request_id="r", phase=StepPhase.PREFILL,
                num_tokens=128, context_len=0,
            )],
            num_prefill_tokens=128, total_scheduled_tokens=128,
            num_prefill_requests=1,
        ),
        deploy,
    )
    plan = QwenModelGraphTemplate(model, ctx=ctx).build_grouped_step(step)
    assert plan.step_id == 42
    assert plan.phase == "prefill"
    assert plan.metadata["model"] == "Qwen3-4B"
    assert plan.metadata["num_layers"] == model.num_layers
    assert plan.metadata["execution_mode"] == "cudagraph"
