"""moe_plan Phase 3.4 验收 + vLLM 默认 EP path 修正: MoE FFN block graph wiring 锁定.

vLLM 默认 EP path 实际不走 AllToAll dispatch/combine, 是 `fused_experts(expert_map=...)`
local compute + 单次 `tensor_model_parallel_all_reduce` 聚合 partial sums. 跟
ep=1 tp>1 path 通信形态完全一样.

锁住:

TP-only (ep == 1, tp > 1) 和 EP (ep > 1, vLLM 默认 path) 共享:
    mlp_norm → moe_gate → moe_topk → moe_dispatch_pre → routed_experts
    → moe_dispatch_post → routed_expert_allreduce (world=max(tp,ep))
    (无 ep_alltoall_dispatch / ep_alltoall_combine)

TRT-LLM SM≥100 / SGLang DeepEP backend 才走 AllToAll (留 follow-up phase, 当前不实现).

外加: MoEDispatch metadata.communication_peer 指向并列 collective op name.
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.engine import build_qwen_roofline_engine
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


def _qwen3_30b_a3b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-30B-A3B",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=0, num_layers=48, vocab_size=151936,
        is_moe=True, num_experts=128, num_activated_experts=8,
        expert_dim=768, num_shared_experts=0,
        moe_layer_freq=1, first_moe_layer=0,
    )


def _moe_layer_ops(*, tp: int, ep: int, tokens: int = 128):
    """Build the MoE layer 0 op list via QwenModelGraphTemplate."""
    model = _qwen3_30b_a3b()
    deploy = DeployConfig(tp_size=tp, ep_size=ep, backend="vllm",
                          backend_version="0.20.1")
    hw = get_hardware_profile("RTX_4090")
    engine = build_qwen_roofline_engine(model, deploy, hw)
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
    step = StepShape.from_workload(wl, deploy)
    grouped = engine.template.build_grouped_step(step)
    layer0_ops = [g.op for g in grouped.groups if g.op.layer_idx == 0]
    return [op.name for op in layer0_ops], layer0_ops


# ---------------------------------------------------------------------------
# TP-only path (ep == 1, tp > 1)
# ---------------------------------------------------------------------------

def test_tp_only_moe_block_order():
    """TP-only: 验 op 顺序 + 无 dispatch alltoall."""
    names, _ = _moe_layer_ops(tp=4, ep=1)
    # 必含 op
    must_have = [
        "mlp_norm", "moe_gate", "moe_topk",
        "moe_dispatch_pre", "routed_experts",
        "moe_dispatch_post", "routed_expert_allreduce",
        "mlp_add",
    ]
    for name in must_have:
        assert name in names, f"TP-only 缺 op: {name} (got {names})"

    # 不含 op (EP-only)
    assert "ep_alltoall_dispatch" not in names
    assert "ep_alltoall_combine" not in names

    # 顺序验证: pre → routed_experts → post → allreduce
    idx_pre = names.index("moe_dispatch_pre")
    idx_routed = names.index("routed_experts")
    idx_post = names.index("moe_dispatch_post")
    idx_ar = names.index("routed_expert_allreduce")
    assert idx_pre < idx_routed < idx_post < idx_ar


def test_tp1_no_routed_allreduce():
    """tp==1 & ep==1: 也无 routed_expert_allreduce (没有 TP 切分)."""
    names, _ = _moe_layer_ops(tp=1, ep=1)
    assert "routed_expert_allreduce" not in names
    # 但 MoEDispatch pre/post 仍存在 (fused_moe local align/sort)
    assert "moe_dispatch_pre" in names
    assert "moe_dispatch_post" in names


# ---------------------------------------------------------------------------
# EP path (ep > 1)
# ---------------------------------------------------------------------------

def test_ep_moe_block_order():
    """vLLM 默认 EP path: 单 AllReduce 跟 TP-only 同, 不走 AllToAll."""
    names, _ = _moe_layer_ops(tp=4, ep=4)
    must_have = [
        "mlp_norm", "moe_gate", "moe_topk",
        "moe_dispatch_pre", "routed_experts", "moe_dispatch_post",
        "routed_expert_allreduce", "mlp_add",
    ]
    for name in must_have:
        assert name in names, f"EP 缺 op: {name} (got {names})"

    # vLLM 默认 path 不走 AllToAll dispatch/combine
    assert "ep_alltoall_dispatch" not in names
    assert "ep_alltoall_combine" not in names

    # 顺序: pre → routed_experts → post → AllReduce
    idx_pre = names.index("moe_dispatch_pre")
    idx_routed = names.index("routed_experts")
    idx_post = names.index("moe_dispatch_post")
    idx_ar = names.index("routed_expert_allreduce")
    assert idx_pre < idx_routed < idx_post < idx_ar


def test_ep_allreduce_world_size_eq_max_tp_ep():
    """vLLM 默认 EP path AllReduce world = max(tp, ep)."""
    _, ops = _moe_layer_ops(tp=4, ep=4)
    ar = next(o for o in ops if o.name == "routed_expert_allreduce")
    assert ar.parallel["world_size"] == 4


# ---------------------------------------------------------------------------
# MoEDispatch metadata 互链
# ---------------------------------------------------------------------------

def test_tp_only_moe_dispatch_metadata_peer():
    """TP-only: moe_dispatch_pre peer = None (ep==1 没 alltoall);
       moe_dispatch_post peer = routed_expert_allreduce (tp>1 才有)."""
    _, ops = _moe_layer_ops(tp=4, ep=1)
    pre = next(o for o in ops if o.name == "moe_dispatch_pre")
    post = next(o for o in ops if o.name == "moe_dispatch_post")
    assert pre.runtime.get("communication_peer") is None
    assert post.runtime.get("communication_peer") == "routed_expert_allreduce"


def test_ep_moe_dispatch_metadata_peer():
    """vLLM 默认 EP path: pre.peer=None (no alltoall), post.peer=routed_expert_allreduce."""
    _, ops = _moe_layer_ops(tp=4, ep=4)
    pre = next(o for o in ops if o.name == "moe_dispatch_pre")
    post = next(o for o in ops if o.name == "moe_dispatch_post")
    assert pre.runtime.get("communication_peer") is None
    assert post.runtime.get("communication_peer") == "routed_expert_allreduce"


# ---------------------------------------------------------------------------
# Op kind/subtype 验证 (plan §3.3 op_subtype convention)
# ---------------------------------------------------------------------------

def test_moe_op_kind_and_subtype():
    """routed_experts op_kind=moe, op_subtype=routed_experts (不再是 fused_moe)."""
    _, ops = _moe_layer_ops(tp=4, ep=1)
    routed = next(o for o in ops if o.name == "routed_experts")
    assert routed.op_kind == "moe"
    assert routed.op_subtype == "routed_experts", (
        f"plan §3.3 要求 op_subtype=routed_experts, got {routed.op_subtype}"
    )
    # kernel_source 仍标 vllm_fused_moe (runtime 属性, 非类名)
    assert routed.runtime["kernel_source"] == "vllm_fused_moe"


def test_moe_dispatch_op_kind_and_subtype():
    """MoEDispatch op_kind=moe_dispatch, op_subtype=pre_dispatch/post_dispatch."""
    _, ops = _moe_layer_ops(tp=4, ep=4)
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
    _, ops = _moe_layer_ops(tp=4, ep=1, tokens=128)
    topk = next(o for o in ops if o.name == "moe_topk")
    assert topk.op_kind == "elementwise"
    assert topk.op_subtype == "topk"
