"""Qwen3Model (dense) 单测 — op_plan §7 + build-once.

锁住:
  - Qwen3-4B prefill / decode 生成 op list
  - per-layer 顺序 + 跨 layer 数量 (representative ops carry layer count)
  - 各 op forward(runtime) 解析出 op_kind / op_subtype / shape / parallel / runtime
  - search-ready: TP 改变后 shape/parallel 跟变, TP>1 生成 dense allreduce

build-once: 图含 prefill+decode 两个 attention 静态 op; 某 step 不活跃的 regime
op.forward(runtime)→None, router 跳过 (active=398, 与旧 per-step 模板等价).
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.step.step_shape import StepShape
from llm_infer_sim.core.models.qwen3 import Qwen3Model
from llm_infer_sim.core.operators import Collective, GEMM
from llm_infer_sim.core.operators.base import Operator
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from llm_infer_sim.core.models.config import ModelConfig
from tests.helpers.support import make_model_config
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload,
    RequestWorkload,
    StepPhase,
)


def _qwen3_4b() -> ModelConfig:
    return make_model_config(
        name="Qwen3-4B",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def _make_ctx(model: ModelConfig, deployment: DeploymentProfile,
              runtime: RuntimeProfile):
    from llm_infer_sim.core.operators.context import build_operator_context
    hw = get_hardware_profile("RTX_4090")
    return build_operator_context(model, deployment, runtime, hw)


def _plan(model: ModelConfig, deployment: DeploymentProfile,
          runtime: RuntimeProfile, step: StepShape):
    return Qwen3Model(
        model=model, ctx=_make_ctx(model, deployment, runtime),
    ).forward(step)


def _resolved(plan):
    """[(op, op_runtime)] for ops active this step (forward()→None ⇒ skipped)."""
    out = []
    for op in plan.ops:
        rt = op.forward(plan.runtime)
        if rt is not None:
            out.append((op, rt))
    return out


def _active_count(plan) -> int:
    return sum(op.count for op in plan.ops if op.forward(plan.runtime) is not None)


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
    return StepShape.from_workload(wl, "eager")


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
    return StepShape.from_workload(wl, "eager")


def test_prefill_op_count():
    """1 embedding + num_layers × 11 per-layer + 1 lm_head (active ops)."""
    model = _qwen3_4b()
    plan = _plan(model, DeploymentProfile.flat(), RuntimeProfile.flat(), _prefill_step())
    expected = 1 + model.num_layers * 11 + 1
    assert _active_count(plan) == expected


def test_decode_op_count():
    model = _qwen3_4b()
    plan = _plan(model, DeploymentProfile.flat(), RuntimeProfile.flat(), _decode_step())
    expected = 1 + model.num_layers * 11 + 1
    assert _active_count(plan) == expected


def test_per_layer_order_first_layer():
    """每层 11 个 active op 顺序: attn_norm/qkv/rope/attn/o/attn_add/mlp_norm/gu/act/down/mlp_add."""
    model = _qwen3_4b()
    plan = _plan(model, DeploymentProfile.flat(), RuntimeProfile.flat(), _prefill_step())
    layer0 = [op for op, _rt in _resolved(plan) if op.layer_idx == 0]
    # 单 attention op 的静态 op_subtype = "attention" (regime 由 forward 解析,
    # prefill step → rt.op_subtype="prefill", 见 test_attention_op_carries_full_shape)。
    expected_subtypes = [
        "attn_norm", "qkv_proj", "rope", "attention",
        "o_proj", "attn_add", "mlp_norm", "gate_up_proj",
        "mlp_act", "down_proj", "mlp_add",
    ]
    assert [op.op_subtype for op in layer0] == expected_subtypes
    for op in layer0:
        assert op.layer_idx == 0


def test_embedding_and_lm_head_are_at_boundaries():
    model = _qwen3_4b()
    plan = _plan(model, DeploymentProfile.flat(), RuntimeProfile.flat(), _prefill_step())
    assert plan.ops[0].op_kind == "embedding"
    assert plan.ops[0].op_subtype == "embedding"
    assert plan.ops[-1].op_kind == "gemm"
    assert plan.ops[-1].op_subtype == "lm_head"


def test_qkv_proj_shape_matches_gqa():
    """Qwen3-4B: num_heads=32, num_kv_heads=8, head_dim=128 → QKV n = (32 + 16) × 128 = 6144."""
    model = _qwen3_4b()
    plan = _plan(model, DeploymentProfile.flat(), RuntimeProfile.flat(), _prefill_step(isl=128))
    op, rt = next((op, rt) for op, rt in _resolved(plan) if op.op_subtype == "qkv_proj")
    assert isinstance(op, GEMM)
    assert rt.shape["m"] == 128
    assert rt.shape["k"] == model.hidden_dim
    # Q dim = 32 × 128 = 4096, K = V = 8 × 128 = 1024; total n = 4096 + 2*1024 = 6144
    assert rt.shape["n"] == 32 * 128 + 2 * 8 * 128
    assert rt.parallel["tp"] == 1
    assert rt.runtime["execution_mode"] == "eager"


def test_attention_op_carries_full_shape():
    """V3 §5.3: attention shape 必带 num_tokens/num_seqs/q_len/kv_len/heads/head_dim."""
    model = _qwen3_4b()
    plan = _plan(model, DeploymentProfile.flat(), RuntimeProfile.flat(), _prefill_step(isl=2048))
    # build-once 图含 prefill+decode 两 regime; prefill step 只 decode 那个 inactive.
    attn = [(op, rt) for op, rt in _resolved(plan) if op.op_kind == "attention"]
    assert len(attn) == 1
    op, rt = attn[0]
    assert op.count == model.num_layers
    assert rt.op_subtype == "prefill"
    assert rt.shape["q_len"] == 2048
    assert rt.shape["kv_len"] == 2048
    assert rt.shape["num_q_heads"] == 32
    assert rt.shape["num_kv_heads"] == 8
    assert rt.shape["head_dim"] == 128
    assert rt.runtime["attention_backend"] == "flash_attn"
    assert rt.runtime["block_size"] == 16


def test_decode_attention_subtype_and_kv_len():
    model = _qwen3_4b()
    plan = _plan(model, DeploymentProfile.flat(), RuntimeProfile.flat(), _decode_step(n=8, ctx=1024))
    attn = [(op, rt) for op, rt in _resolved(plan) if op.op_kind == "attention"]
    assert len(attn) == 1
    _op, rt = attn[0]
    assert rt.op_subtype == "decode"
    assert rt.shape["q_len"] == 1
    assert rt.shape["kv_len"] == 1024
    assert rt.shape["num_seqs"] == 8


def test_lm_head_tokens_equals_num_requests():
    """prefill: 每 req 1 个采样 token; decode: 同."""
    model = _qwen3_4b()
    plan_p = _plan(model, DeploymentProfile.flat(), RuntimeProfile.flat(), _prefill_step(isl=2048))
    _op, rt_p = next((op, rt) for op, rt in _resolved(plan_p) if op.op_subtype == "lm_head")
    assert rt_p.shape["m"] == 1
    plan_d = _plan(model, DeploymentProfile.flat(), RuntimeProfile.flat(), _decode_step(n=8))
    _op, rt_d = next((op, rt) for op, rt in _resolved(plan_d) if op.op_subtype == "lm_head")
    assert rt_d.shape["m"] == 8


def test_tp_affects_qkv_shape():
    """search-ready: TP 改变后 GEMM n 维按 tp 切."""
    model = _qwen3_4b()
    step = _prefill_step(isl=128)
    plan_tp1 = _plan(model, DeploymentProfile.flat(tp=1), RuntimeProfile.flat(), step)
    plan_tp2 = _plan(model, DeploymentProfile.flat(tp=2), RuntimeProfile.flat(), step)
    _o1, qkv1 = next((op, rt) for op, rt in _resolved(plan_tp1) if op.op_subtype == "qkv_proj")
    _o2, qkv2 = next((op, rt) for op, rt in _resolved(plan_tp2) if op.op_subtype == "qkv_proj")
    # tp=2 后 num_q_heads_per_tp=16, num_kv_heads_per_tp=4 → n = (16+8)*128 = 3072 = qkv1.n/2
    assert qkv2.shape["n"] == qkv1.shape["n"] // 2
    assert qkv1.parallel["tp"] == 1
    assert qkv2.parallel["tp"] == 2


def test_tp1_has_no_dense_allreduce():
    """tp==1: dense allreduce op 仍在 build-once 图里 (结构 tp-无关), 但 world_size=1
    → forward()->None → 不进 active/trace, 无通信成本."""
    model = _qwen3_4b()
    plan = _plan(model, DeploymentProfile.flat(tp=1), RuntimeProfile.flat(), _prefill_step(isl=128))
    by_name = {op.name: op for op in plan.ops}
    for name in ("tp_o_proj_allreduce", "tp_down_proj_allreduce"):
        assert name in by_name                              # build-once: 结构存在
        assert by_name[name].parallel["world_size"] == 1
        assert by_name[name].forward(plan.runtime) is None  # 运行时 inactive
    # active (router 实际估算) 里不含 allreduce
    active_names = {op.name for op, _rt in _resolved(plan)}
    assert "tp_o_proj_allreduce" not in active_names
    assert "tp_down_proj_allreduce" not in active_names


def test_tp_dense_allreduce_grouped_shape_and_bytes():
    model = _qwen3_4b()
    deployment = DeploymentProfile.flat(tp=2)
    runtime = RuntimeProfile.flat()
    ctx = _make_ctx(model, deployment, runtime)
    step = _prefill_step(isl=128)
    plan = Qwen3Model(model=model, ctx=ctx).forward(step)

    resolved_by_name = {op.name: (op, rt) for op, rt in _resolved(plan)}
    expected_bytes = int(step.total_tokens * model.hidden_dim * ctx.a_byte)
    for name in ("tp_o_proj_allreduce", "tp_down_proj_allreduce"):
        op, rt = resolved_by_name[name]
        assert isinstance(op, Collective)
        assert op.op_subtype == "allreduce"
        assert op.roofline_spec(rt).comm_bytes == expected_bytes
        assert op.parallel["world_size"] == 2
        assert op.count == model.num_layers


def test_all_ops_have_required_metadata():
    """所有 active op 经 forward(runtime) 必含 op_kind/op_subtype/shape/parallel/runtime."""
    model = _qwen3_4b()
    plan = _plan(model, DeploymentProfile.flat(), RuntimeProfile.flat(), _prefill_step())
    for op, rt in _resolved(plan):
        assert isinstance(op, Operator)
        assert op.op_kind and op.op_subtype
        assert rt.shape, f"{op.name} no shape"
        assert rt.parallel, f"{op.name} no parallel"
        assert rt.runtime, f"{op.name} no runtime"
        assert op.roofline_spec(rt), f"{op.name} no roofline_spec"


def test_plan_metadata_carries_model_info():
    model = _qwen3_4b()
    deployment = DeploymentProfile.flat()
    runtime = RuntimeProfile.flat(execution_mode="cudagraph")
    ctx = _make_ctx(model, deployment, runtime)
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
        runtime.execution.execution_mode,
    )
    plan = Qwen3Model(model=model, ctx=ctx).forward(step)
    assert plan.step_id == 42
    assert plan.phase == "prefill"
    assert plan.metadata["model"] == "Qwen3-4B"
    assert plan.metadata["num_layers"] == model.num_layers
    assert plan.metadata["execution_mode"] == "cudagraph"
