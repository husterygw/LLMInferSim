"""Expert Parallelism (EP) cost 公式一致性.

链路: QwenModelGraphTemplate._build_moe_ffn_block 直接构造 FusedMoE + Collective (ep_alltoall_dispatch / combine).

固化以下:
  1. ep>1 时 expert_dim_per_device = expert_dim (不切 tp), 跟 ep=1 时反着
  2. routed_experts.flops = tokens × top_k × 3 × 2 × h × expert_dim // ep
  3. weight = distinct(T,k,N) × 3 × h × expert_dim × w_byte / ep
  4. ep>1 时 ep_alltoall_dispatch + combine 同时出现, comm_bytes = tokens × h × a_byte
  5. ep>1 时 routed_expert_allreduce 消失
  6. Collective.parallel.world_size = ep (跟 DP+EP 场景 ep = tp × dp 一致)
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.engine import build_qwen_roofline_engine
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.operators import estimate_distinct_experts
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


W_BYTE = 2.0
A_BYTE = 2.0


def _qwen3_30b_a3b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-30B-A3B",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=0, num_layers=48, vocab_size=151936,
        is_moe=True, num_experts=128, num_activated_experts=8,
        expert_dim=768, num_shared_experts=0,
        moe_layer_freq=1, first_moe_layer=0,
    )


def _layer_ops(*, tokens: int, phase: str, tp: int, ep: int,
               layer_idx: int = 0, ctx: int = 128):
    model = _qwen3_30b_a3b()
    deploy = DeployConfig(tp_size=tp, ep_size=ep)
    hw = get_hardware_profile("RTX_4090")
    engine = build_qwen_roofline_engine(model, deploy, hw)

    if phase == "prefill":
        wl = GlobalStepWorkload(
            step_id=0, phase=StepPhase.PREFILL,
            requests=[RequestWorkload(
                request_id="r0", phase=StepPhase.PREFILL,
                num_tokens=tokens, context_len=0,
            )],
            num_prefill_tokens=tokens, total_scheduled_tokens=tokens,
            num_prefill_requests=1,
        )
    else:
        wl = GlobalStepWorkload(
            step_id=0, phase=StepPhase.DECODE,
            requests=[
                RequestWorkload(
                    request_id=f"d{i}", phase=StepPhase.DECODE,
                    num_tokens=1, context_len=ctx,
                )
                for i in range(tokens)
            ],
            num_decode_tokens=tokens, total_scheduled_tokens=tokens,
            num_decode_requests=tokens,
        )
    step = StepShape.from_workload(wl, deploy)
    return engine.template._build_moe_layer(layer_idx, step)


def _find(ops, name):
    for op in ops:
        if op.name == name:
            return op
    raise AssertionError(f"op {name!r} not found in {[o.name for o in ops]}")


def _has(ops, name) -> bool:
    return any(op.name == name for op in ops)


# ------- routed_experts under EP -------

def test_routed_experts_expert_dim_not_sliced_under_ep():
    """ep>1 时 expert_dim_per_device = expert_dim (不切 tp), 跟 ep=1 时反着."""
    m = _qwen3_30b_a3b()
    tokens = 4
    ep = 2
    ops = _layer_ops(tokens=tokens, phase="decode", tp=2, ep=ep)
    op = _find(ops, "routed_experts")

    expert_dim_per_device = m.expert_dim   # NOT // tp
    expected_flops = (
        tokens * m.num_activated_experts * 3 * 2 * m.hidden_dim
        * expert_dim_per_device // ep
    )
    assert op.roofline_spec().flops == expected_flops


def test_routed_experts_weight_uses_distinct_div_ep():
    """EP 下 weight read = distinct × per-expert / ep."""
    m = _qwen3_30b_a3b()
    ep = 2
    expert_dim_per_device = m.expert_dim   # ep>1 不切 tp

    for tokens in (4, 128):
        ops = _layer_ops(tokens=tokens, phase="prefill", tp=2, ep=ep)
        op = _find(ops, "routed_experts")
        distinct = estimate_distinct_experts(
            tokens, m.num_activated_experts, m.num_experts, skew=0.0,
        )
        expected_weight = int(
            distinct * 3 * m.hidden_dim * expert_dim_per_device
            * W_BYTE / ep
        )
        tokens_per_device = tokens * m.num_activated_experts // ep
        expected_act = 2 * tokens_per_device * m.hidden_dim * A_BYTE
        assert op.roofline_spec().mem_bytes == expected_weight + expected_act, (
            f"tokens={tokens}: distinct={distinct:.2f}"
        )


def test_routed_experts_act_scales_with_tokens_per_device():
    """EP 下 act_in = act_out = (tokens × top_k // ep) × h × a_byte."""
    m = _qwen3_30b_a3b()
    tokens = 4
    ep = 2
    ops = _layer_ops(tokens=tokens, phase="decode", tp=2, ep=ep)
    op = _find(ops, "routed_experts")
    f = op.roofline_spec()

    tokens_per_device = tokens * m.num_activated_experts // ep
    expected_act_each = tokens_per_device * m.hidden_dim * A_BYTE
    assert f.load_act == expected_act_each
    assert f.store_act == expected_act_each


# ------- AllToAll comm injection -------

def test_ep_alltoall_dispatch_combine_both_present():
    """ep>1 时必须同时插入 dispatch + combine 两个 AllToAll ops."""
    ops = _layer_ops(tokens=4, phase="decode", tp=2, ep=2)
    assert _has(ops, "ep_alltoall_dispatch")
    assert _has(ops, "ep_alltoall_combine")


def test_ep_alltoall_comm_bytes():
    """AllToAll comm_bytes = tokens × h × a_byte (per rank 发/收量)."""
    m = _qwen3_30b_a3b()
    ep = 2

    for tokens in (4, 128):
        phase = "prefill" if tokens > 1 else "decode"
        ops = _layer_ops(tokens=tokens, phase=phase, tp=2, ep=ep)
        for op_name in ("ep_alltoall_dispatch", "ep_alltoall_combine"):
            op = _find(ops, op_name)
            f = op.roofline_spec()
            expected = tokens * m.hidden_dim * A_BYTE
            assert f.comm_bytes == expected, (
                f"tokens={tokens} op={op_name}"
            )
            assert f.comm_type == "alltoall"


def test_alltoall_world_size_equals_ep():
    """Collective.parallel.world_size 应严格等于 ep_size (不是 tp)."""
    for ep in (2, 4):
        ops = _layer_ops(tokens=4, phase="decode", tp=2, ep=ep)
        for name in ("ep_alltoall_dispatch", "ep_alltoall_combine"):
            op = _find(ops, name)
            assert op.parallel["world_size"] == ep, f"ep={ep} {name}"


# ------- routed_expert_allreduce gone under EP -------

def test_routed_expert_allreduce_absent_under_ep():
    """ep>1 时 routed_expert_allreduce 被 AllToAll 替代, 不应再出现."""
    ops = _layer_ops(tokens=4, phase="decode", tp=2, ep=2)
    assert not _has(ops, "routed_expert_allreduce")


# ------- ep=1 vs ep>1 切换 -------

def test_ep1_uses_allreduce_ep2_uses_alltoall():
    """同 model 在 ep=1 / ep=2 之间切换, 通信 op 形态正确切换."""
    ops_ep1 = _layer_ops(tokens=4, phase="decode", tp=2, ep=1)
    ops_ep2 = _layer_ops(tokens=4, phase="decode", tp=2, ep=2)

    assert _has(ops_ep1, "routed_expert_allreduce")
    assert not _has(ops_ep1, "ep_alltoall_dispatch")
    assert not _has(ops_ep1, "ep_alltoall_combine")

    assert not _has(ops_ep2, "routed_expert_allreduce")
    assert _has(ops_ep2, "ep_alltoall_dispatch")
    assert _has(ops_ep2, "ep_alltoall_combine")


def test_dp_doubles_ep_world_size():
    """DP+EP 场景: tp=2 dp=2 ep=4 vs tp=2 dp=1 ep=2.
    Collective.parallel.world_size 必须等于 ep_size, 不是 tp."""
    ops_small = _layer_ops(tokens=4, phase="decode", tp=2, ep=2)
    ops_big = _layer_ops(tokens=4, phase="decode", tp=2, ep=4)

    dispatch_small = _find(ops_small, "ep_alltoall_dispatch")
    dispatch_big = _find(ops_big, "ep_alltoall_dispatch")
    assert dispatch_small.parallel["world_size"] == 2
    assert dispatch_big.parallel["world_size"] == 4
