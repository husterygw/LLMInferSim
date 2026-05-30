"""MoE FFN block graph wiring 锁定 (AIC 对齐: MoEDispatch=通信, MoE=计算).

vLLM 默认 MoE 路径: `fused_experts(expert_map=...)` 本地计算 + 单次
`tensor_model_parallel_all_reduce` 聚合 partial sums (TP / EP 通信形态相同).
通信由 moe_dispatch_post 承载 (forward 按 tp/ep 解析为 allreduce(max(tp,ep))
或 None=本地). 不再有独立 routed_expert_allreduce op.

锁住的结构:
    mlp_norm → moe_gate → moe_topk → moe_dispatch_pre → routed_experts
    → moe_dispatch_post (= 跨卡 allreduce when max(tp,ep)>1, 否则本地) → mlp_add

TRT-LLM SM≥100 / SGLang DeepEP backend 走 AllToAll (留 follow-up phase, 当前不实现).
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.engine import build_qwen_roofline_engine
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from llm_infer_sim.core.models.config import ModelConfig
from tests.helpers.support import make_model_config
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


def _qwen3_30b_a3b() -> ModelConfig:
    return make_model_config(
        name="Qwen3-30B-A3B",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=0, num_layers=48, vocab_size=151936,
        is_moe=True, num_experts=128, num_activated_experts=8,
        expert_dim=768, num_shared_experts=0,
        moe_layer_freq=1, first_moe_layer=0,
    )


def _moe_layer_ops(*, tp: int, ep: int, tokens: int = 128):
    """Build the MoE layer 0 op list via the Qwen engine template."""
    model = _qwen3_30b_a3b()
    deployment = DeploymentProfile.flat(tp=tp, ep=ep)
    runtime = RuntimeProfile.flat(backend="vllm", backend_version="0.20.1")
    hw = get_hardware_profile("RTX_4090")
    engine = build_qwen_roofline_engine(model, deployment, runtime, hw)
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=tokens, context_len=0,
        )],
        num_prefill_tokens=tokens, total_scheduled_tokens=tokens,
        num_prefill_requests=1,
    )
    from llm_infer_sim.core.graph.step_shape import StepShape
    step = StepShape.from_workload(wl, runtime.execution.execution_mode)
    plan = engine.model.forward(step)
    layer0_ops = [op for op in plan.ops if op.layer_idx == 0]
    return [op.name for op in layer0_ops], layer0_ops, plan.runtime


# ---------------------------------------------------------------------------
# TP-only path (ep == 1, tp > 1)
# ---------------------------------------------------------------------------

def test_tp_only_moe_block_order():
    """TP-only: 验 MoE block op 顺序 (跨卡 allreduce 现由 moe_dispatch_post 承载)."""
    names, _, _ = _moe_layer_ops(tp=4, ep=1)
    must_have = [
        "mlp_norm", "moe_gate", "moe_topk",
        "moe_dispatch_pre", "routed_experts",
        "moe_dispatch_post", "mlp_add",
    ]
    for name in must_have:
        assert name in names, f"TP-only 缺 op: {name} (got {names})"

    # 通信折进 moe_dispatch_post, 不再有独立 routed_expert_allreduce
    assert "routed_expert_allreduce" not in names
    # 不走 AllToAll (vLLM 默认非 DeepEP)
    assert "ep_alltoall_dispatch" not in names
    assert "ep_alltoall_combine" not in names

    # 顺序: pre → routed_experts → post
    idx_pre = names.index("moe_dispatch_pre")
    idx_routed = names.index("routed_experts")
    idx_post = names.index("moe_dispatch_post")
    assert idx_pre < idx_routed < idx_post


def test_tp1_no_routed_allreduce():
    """tp==1 & ep==1: moe_dispatch_post 解析为本地 (无跨卡通信),
    无独立 routed_expert_allreduce op."""
    names, ops, runtime = _moe_layer_ops(tp=1, ep=1)
    assert "routed_expert_allreduce" not in names    # 已折进 moe_dispatch_post
    assert "moe_dispatch_pre" in names
    assert "moe_dispatch_post" in names
    # post-dispatch forward 解析为本地 (op_subtype=None, world_size=1)
    post = next(o for o in ops if o.name == "moe_dispatch_post")
    rt = post.forward(runtime)
    assert rt.op_subtype is None                     # 无 collective
    assert rt.parallel["world_size"] == 1
    assert rt.shape["message_bytes"] == 0


# ---------------------------------------------------------------------------
# EP path (ep > 1)
# ---------------------------------------------------------------------------

def test_ep_moe_block_order():
    """vLLM 默认 EP path: 通信由 moe_dispatch_post 承载 (allreduce), 不走 AllToAll."""
    names, _, _ = _moe_layer_ops(tp=4, ep=4)
    must_have = [
        "mlp_norm", "moe_gate", "moe_topk",
        "moe_dispatch_pre", "routed_experts", "moe_dispatch_post", "mlp_add",
    ]
    for name in must_have:
        assert name in names, f"EP 缺 op: {name} (got {names})"

    assert "routed_expert_allreduce" not in names    # 折进 moe_dispatch_post
    assert "ep_alltoall_dispatch" not in names
    assert "ep_alltoall_combine" not in names

    idx_pre = names.index("moe_dispatch_pre")
    idx_routed = names.index("routed_experts")
    idx_post = names.index("moe_dispatch_post")
    assert idx_pre < idx_routed < idx_post


def test_ep_allreduce_world_size_eq_max_tp_ep():
    """moe_dispatch_post forward 解析: vLLM 默认 EP path → allreduce(world=max(tp,ep))."""
    _, ops, runtime = _moe_layer_ops(tp=4, ep=4)
    post = next(o for o in ops if o.name == "moe_dispatch_post")
    rt = post.forward(runtime)
    assert rt.op_subtype == "allreduce"
    assert rt.parallel["world_size"] == 4


# ---------------------------------------------------------------------------
# MoEDispatch metadata 互链
# ---------------------------------------------------------------------------

def test_tp_only_moe_dispatch_comm_resolution():
    """TP-only: pre 本地 (无通信); post 承载 allreduce(world=tp)."""
    _, ops, runtime = _moe_layer_ops(tp=4, ep=1)
    pre = next(o for o in ops if o.name == "moe_dispatch_pre")
    post = next(o for o in ops if o.name == "moe_dispatch_post")
    assert pre.forward(runtime).op_subtype is None        # pre 永远本地
    rt_post = post.forward(runtime)
    assert rt_post.op_subtype == "allreduce"
    assert rt_post.parallel["world_size"] == 4


def test_ep_moe_dispatch_comm_resolution():
    """vLLM 默认 EP path (非 DeepEP): post = allreduce(world=max(tp,ep))."""
    _, ops, runtime = _moe_layer_ops(tp=4, ep=4)
    pre = next(o for o in ops if o.name == "moe_dispatch_pre")
    post = next(o for o in ops if o.name == "moe_dispatch_post")
    assert pre.forward(runtime).op_subtype is None
    rt_post = post.forward(runtime)
    assert rt_post.op_subtype == "allreduce"
    assert rt_post.parallel["world_size"] == 4


# ---------------------------------------------------------------------------
# Op kind/subtype 验证 (plan §3.3 op_subtype convention)
# ---------------------------------------------------------------------------

def test_moe_op_kind_and_subtype():
    """routed_experts op_kind=moe, op_subtype=routed_experts (不再是 fused_moe)."""
    _, ops, _ = _moe_layer_ops(tp=4, ep=1)
    routed = next(o for o in ops if o.name == "routed_experts")
    assert routed.op_kind == "moe"
    assert routed.op_subtype == "routed_experts", (
        f"plan §3.3 要求 op_subtype=routed_experts, got {routed.op_subtype}"
    )
    # kernel_source 仍标 vllm_fused_moe (runtime 属性, 非类名)
    assert routed.runtime["kernel_source"] == "vllm_fused_moe"


def test_moe_dispatch_op_kind_and_subtype():
    """MoEDispatch op_kind=moe_dispatch, op_subtype=pre_dispatch/post_dispatch."""
    _, ops, _ = _moe_layer_ops(tp=4, ep=4)
    pre = next(o for o in ops if o.name == "moe_dispatch_pre")
    post = next(o for o in ops if o.name == "moe_dispatch_post")
    assert pre.op_kind == "moe_dispatch"
    assert pre.op_subtype == "pre_dispatch"
    assert post.op_kind == "moe_dispatch"
    assert post.op_subtype == "post_dispatch"


# ---------------------------------------------------------------------------
# moe_topk shape (新加 op)
# ---------------------------------------------------------------------------

def test_moe_topk_present_in_block():
    """plan §3.3 op#3: 显式 moe_topk op (softmax + topk) 必须存在."""
    _, ops, _ = _moe_layer_ops(tp=4, ep=1, tokens=128)
    topk = next(o for o in ops if o.name == "moe_topk")
    assert topk.op_kind == "elementwise"
    assert topk.op_subtype == "topk"
