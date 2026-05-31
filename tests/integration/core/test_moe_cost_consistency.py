"""MoE cost 公式数字一致性 (V3 §5.2 + IMPL_PLAN §4 routed_experts 语义).

链路: common.build_moe_ffn_block 构造 build-once 静态 op (GEMM moe_gate +
FusedMoE routed_experts + Collective ep/tp 通信 op); op.forward(runtime) 解析
step 形状, roofline_spec(op_runtime) 重算 (recompute==baked, 见 test_moe_static_contract).

固化以下关键正确性:
  1. routed_experts.flops = tokens × top_k × 3 × 2 × h × expert_dim_per_device
  2. routed_experts.mem_bytes 的 weight 部分只读 top_k 个 expert (decode 边界)
  3. routed_experts.mem_bytes 不含中间激活的 HBM 读写 (FusedMoE 语义)
  4. moe_gate.flops = 2 × tokens × h × num_experts
  5. ep=1 + tp>1: 跨卡 allreduce 由 moe_dispatch_post 承载 (AIC 对齐, 无独立 allreduce op)
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.operators import (
    MoERoutingProfile,
    estimate_distinct_experts,
)
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.step.step_shape import StepShape
from llm_infer_sim.core.models.qwen3_moe import Qwen3MoeModel
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


# BF16 unquantized (跟 OperatorContext 默认对齐)
W_BYTE = 2.0
A_BYTE = 2.0


def _qwen3_30b_a3b() -> ModelConfig:
    return make_model_config(
        name="Qwen3-30B-A3B",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=0, num_layers=48, vocab_size=151936,
        is_moe=True, num_experts=128, num_activated_experts=8,
        expert_dim=768, num_shared_experts=0,
        moe_layer_freq=1, first_moe_layer=0,
    )


def _layer_ops(*, tokens: int, phase: str, tp: int = 2, ep: int = 1,
               layer_idx: int = 0, routing: MoERoutingProfile | None = None,
               ctx: int = 128):
    """Production Qwen3MoeModel graph ops + step runtime. Returns (ops, runtime).

    Builds the real model graph (build-once) and returns the full op list; tests
    pick MoE ops by name via _find. Specs are recomputed per-step via
    op.forward(runtime); use _spec(op, rt)."""
    model = _qwen3_30b_a3b()
    deployment = DeploymentProfile.flat(tp=tp, ep=ep)
    runtime = RuntimeProfile.flat()
    hw = get_hardware_profile("RTX_4090")
    routing = routing or MoERoutingProfile.balanced()
    octx = build_operator_context(model, deployment, runtime, hw, routing=routing)

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
    step = StepShape.from_workload(wl, "eager")
    plan = Qwen3MoeModel(model_config=model, ctx=octx).forward(step)
    return plan.ops, plan.runtime


def _spec(op, runtime):
    """Step-resolved roofline spec (build-once op + forward)."""
    return op.roofline_spec(op.forward(runtime))


def _find(ops, name):
    for op in ops:
        if op.name == name:
            return op
    raise AssertionError(f"op {name!r} not found in {[o.name for o in ops]}")


def _has(ops, name) -> bool:
    return any(op.name == name for op in ops)


# ---- routed_experts FLOPs ----

def test_routed_experts_flops_uses_top_k_not_num_experts():
    """每 token 只对 top_k 个 expert 算 FFN, 不是全部 128 个."""
    m = _qwen3_30b_a3b()
    tokens = 4
    ops, rt = _layer_ops(tokens=tokens, phase="decode")
    op = _find(ops, "routed_experts")

    expert_dim_per_device = m.expert_dim // 2     # tp=2
    expected_flops = (
        tokens * m.num_activated_experts * 3 * 2 * m.hidden_dim * expert_dim_per_device
    )
    assert _spec(op, rt).flops == expected_flops

    wrong = tokens * m.num_experts * 3 * 2 * m.hidden_dim * expert_dim_per_device
    assert _spec(op, rt).flops != wrong


def test_routed_experts_weight_read_decode_single_token_equals_topk():
    """tokens=1 + skew=0: distinct == top_k (decode 严格)."""
    m = _qwen3_30b_a3b()
    tokens = 1
    ops, rt = _layer_ops(tokens=tokens, phase="decode")
    op = _find(ops, "routed_experts")

    expert_dim_per_device = m.expert_dim // 2
    expected_weight = int(
        m.num_activated_experts * 3 * m.hidden_dim * expert_dim_per_device * W_BYTE
    )
    expected_act = 2 * tokens * m.hidden_dim * A_BYTE
    assert _spec(op, rt).mem_bytes == expected_weight + expected_act


def test_routed_experts_weight_read_scales_with_distinct():
    """tokens > 1 + skew=0: weight read = distinct(coupon collector) × 3 × h × dim × w_byte."""
    m = _qwen3_30b_a3b()
    expert_dim_per_device = m.expert_dim // 2

    for tokens in (4, 128):
        ops, rt = _layer_ops(tokens=tokens, phase="prefill")
        op = _find(ops, "routed_experts")
        distinct = estimate_distinct_experts(
            tokens, m.num_activated_experts, m.num_experts, skew=0.0,
        )
        expected_weight = int(
            distinct * 3 * m.hidden_dim * expert_dim_per_device * W_BYTE
        )
        expected_act = 2 * tokens * m.hidden_dim * A_BYTE
        assert _spec(op, rt).mem_bytes == expected_weight + expected_act, (
            f"tokens={tokens}: distinct={distinct:.2f}"
        )


def test_routed_experts_no_intermediate_hbm_write():
    """FusedMoE 中间激活不写 HBM. tokens=1 严格 weight + Q/O."""
    m = _qwen3_30b_a3b()
    tokens = 1
    ops, rt = _layer_ops(tokens=tokens, phase="decode")
    op = _find(ops, "routed_experts")

    expert_dim_per_device = m.expert_dim // 2
    weight = m.num_activated_experts * 3 * m.hidden_dim * expert_dim_per_device * W_BYTE
    qo_act = 2 * tokens * m.hidden_dim * A_BYTE
    assert _spec(op, rt).mem_bytes == int(weight + qo_act)

    # 防御: 若误加 naive intermediate 项 mem_bytes 会变大
    m_e = max(1, tokens * m.num_activated_experts // m.num_experts)
    naive_intermediate = 2 * m_e * 2 * expert_dim_per_device * A_BYTE
    assert _spec(op, rt).mem_bytes < weight + qo_act + naive_intermediate


def test_moe_gate_flops():
    """Router gate 是全连接 GEMM: flops = 2 × tokens × hidden × num_experts."""
    m = _qwen3_30b_a3b()
    tokens = 4
    ops, rt = _layer_ops(tokens=tokens, phase="decode")
    op = _find(ops, "moe_gate")
    assert _spec(op, rt).flops == 2 * tokens * m.hidden_dim * m.num_experts


def test_moe_dispatch_post_carries_allreduce_under_ep1_tp2():
    """ep=1 + tp>1 (Row Parallel 风格): 通信由 moe_dispatch_post 承载
    (AIC 对齐, 不再有独立 routed_expert_allreduce op)."""
    m = _qwen3_30b_a3b()
    tokens = 4
    ops, rt = _layer_ops(tokens=tokens, phase="decode", tp=2, ep=1)
    assert not _has(ops, "routed_expert_allreduce")     # 已折进 dispatch_post
    post = _find(ops, "moe_dispatch_post")
    op_rt = post.forward(rt)
    assert op_rt.op_subtype == "allreduce"
    assert op_rt.parallel["world_size"] == 2
    assert op_rt.shape["message_bytes"] == tokens * m.hidden_dim * A_BYTE


def test_routed_experts_storage_vs_read_ratio_matches_topk_for_single_token():
    """关键 MoE 收益: 单 token (decode) 读字节/总存储 ≈ top_k / num_experts (= 8/128 = 0.0625)."""
    m = _qwen3_30b_a3b()
    tp = 2
    expert_dim_per_device = m.expert_dim // tp
    layer_storage_bytes = (
        m.num_experts * 3 * m.hidden_dim * expert_dim_per_device * W_BYTE
    )
    per_token_read = (
        m.num_activated_experts * 3 * m.hidden_dim * expert_dim_per_device * W_BYTE
    )
    ratio = per_token_read / layer_storage_bytes
    expected_ratio = m.num_activated_experts / m.num_experts
    assert ratio == pytest.approx(expected_ratio)


def test_skew_one_pins_routed_experts_back_to_topk():
    """skew=1 (极端 imbalance): weight_read 退化到 top_k 公式 (worst-case)."""
    m = _qwen3_30b_a3b()
    tokens = 128
    routing = MoERoutingProfile(distribution="balanced", skew=1.0)
    ops, rt = _layer_ops(tokens=tokens, phase="prefill", routing=routing)
    op = _find(ops, "routed_experts")

    expert_dim_per_device = m.expert_dim // 2
    expected_weight = int(
        m.num_activated_experts * 3 * m.hidden_dim * expert_dim_per_device * W_BYTE
    )
    expected_act = 2 * tokens * m.hidden_dim * A_BYTE
    assert _spec(op, rt).mem_bytes == expected_weight + expected_act
