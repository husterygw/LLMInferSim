"""Expert Parallelism (EP) cost 公式一致性.

链路: Qwen3MoeModel.forward 构造 build-once MoE 图; op.forward(runtime) 解析 step,
roofline_spec(op_runtime) 重算 (recompute==baked).

固化以下:
  1. ep>1 时 expert_dim_per_device = expert_dim (不切 tp), 跟 ep=1 时反着
  2. routed_experts.flops = tokens × top_k × 3 × 2 × h × expert_dim // ep
  3. weight = distinct(T,k,N) × 3 × h × expert_dim × w_byte / ep
  4. 跨卡通信由 moe_dispatch_post 承载 (AIC 对齐): vLLM 默认 → allreduce(max(tp,ep)),
     不走 all2all; 无独立 routed_expert_allreduce op
  5. moe_dispatch_post.forward.parallel.world_size = max(tp, ep)
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.operators import MoERoutingProfile, estimate_distinct_experts
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.models.qwen3_moe import Qwen3MoeModel
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from llm_infer_sim.core.models.config import ModelConfig
from tests.helpers.support import make_model_config
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


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


def _layer_ops(*, tokens: int, phase: str, tp: int, ep: int,
               layer_idx: int = 0, ctx: int = 128):
    """Production Qwen3MoeModel graph ops + step runtime. Returns (ops, runtime).

    Tests pick MoE ops by name via _find from the full model op list."""
    model = _qwen3_30b_a3b()
    deployment = DeploymentProfile.flat(tp=tp, ep=ep)
    runtime = RuntimeProfile.flat()
    hw = get_hardware_profile("RTX_4090")
    octx = build_operator_context(model, deployment, runtime, hw,
                                  routing=MoERoutingProfile.balanced())

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
    return op.roofline_spec(op.forward(runtime))


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
    ops, rt = _layer_ops(tokens=tokens, phase="decode", tp=2, ep=ep)
    op = _find(ops, "routed_experts")

    expert_dim_per_device = m.expert_dim   # NOT // tp
    expected_flops = (
        tokens * m.num_activated_experts * 3 * 2 * m.hidden_dim
        * expert_dim_per_device // ep
    )
    assert _spec(op, rt).flops == expected_flops


def test_routed_experts_weight_uses_distinct_div_ep():
    """EP 下 weight read = distinct × per-expert / ep."""
    m = _qwen3_30b_a3b()
    ep = 2
    expert_dim_per_device = m.expert_dim   # ep>1 不切 tp

    for tokens in (4, 128):
        ops, rt = _layer_ops(tokens=tokens, phase="prefill", tp=2, ep=ep)
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
        assert _spec(op, rt).mem_bytes == expected_weight + expected_act, (
            f"tokens={tokens}: distinct={distinct:.2f}"
        )


def test_routed_experts_act_scales_with_tokens_per_device():
    """EP 下 act_in = act_out = (tokens × top_k // ep) × h × a_byte."""
    m = _qwen3_30b_a3b()
    tokens = 4
    ep = 2
    ops, rt = _layer_ops(tokens=tokens, phase="decode", tp=2, ep=ep)
    op = _find(ops, "routed_experts")
    f = _spec(op, rt)

    tokens_per_device = tokens * m.num_activated_experts // ep
    expected_act_each = tokens_per_device * m.hidden_dim * A_BYTE
    assert f.load_act == expected_act_each
    assert f.store_act == expected_act_each


# ------- vLLM 默认 EP path: 单 AllReduce, 不 AllToAll -------
# (TRT-LLM SM≥100 / SGLang DeepEP 才走 AllToAll dispatch/combine; vLLM 默认
#  `fused_experts(expert_map=...) + tensor_model_parallel_all_reduce` 单次 AllReduce
#  跨 max(tp, ep) ranks 聚合 partial sums.)

def test_ep_uses_single_allreduce_not_alltoall():
    """vLLM 默认 EP path: 通信由 moe_dispatch_post 承载 allreduce, 不走 AllToAll;
    不再有独立 routed_expert_allreduce op."""
    ops, rt = _layer_ops(tokens=4, phase="decode", tp=2, ep=2)
    assert not _has(ops, "ep_alltoall_dispatch")
    assert not _has(ops, "ep_alltoall_combine")
    assert not _has(ops, "routed_expert_allreduce")
    post = _find(ops, "moe_dispatch_post")
    assert post.forward(rt).op_subtype == "allreduce"


def test_ep_allreduce_comm_bytes():
    """post-dispatch comm bytes = tokens × hidden × a_byte (output 聚合, EP/TP 同公式)."""
    m = _qwen3_30b_a3b()
    for tokens in (4, 128):
        phase = "prefill" if tokens > 1 else "decode"
        ops, rt = _layer_ops(tokens=tokens, phase=phase, tp=2, ep=2)
        op_rt = _find(ops, "moe_dispatch_post").forward(rt)
        expected = tokens * m.hidden_dim * A_BYTE
        assert op_rt.shape["message_bytes"] == expected, (
            f"tokens={tokens}: got {op_rt.shape['message_bytes']} != {expected}"
        )
        assert op_rt.op_subtype == "allreduce"


def test_allreduce_world_size_equals_max_tp_ep():
    """post-dispatch world_size = max(tp, ep) (vLLM default = world group when EP enabled)."""
    for tp, ep, expected_ws in [(2, 2, 2), (4, 1, 4), (2, 4, 4), (4, 2, 4)]:
        ops, rt = _layer_ops(tokens=4, phase="decode", tp=tp, ep=ep)
        op_rt = _find(ops, "moe_dispatch_post").forward(rt)
        assert op_rt.parallel["world_size"] == expected_ws, (
            f"tp={tp} ep={ep}: world_size={op_rt.parallel['world_size']} != {expected_ws}"
        )


def test_ep1_and_ep2_both_use_allreduce():
    """tp>1 时 post-dispatch 在 ep=1 和 ep>1 都解析为 allreduce (vLLM 默认 path)."""
    for label, ep in [("ep=1", 1), ("ep=2", 2)]:
        ops, rt = _layer_ops(tokens=4, phase="decode", tp=2, ep=ep)
        assert not _has(ops, "routed_expert_allreduce"), f"{label} 不应有独立 allreduce"
        assert not _has(ops, "ep_alltoall_dispatch"), f"{label} 不应有 alltoall"
        assert not _has(ops, "ep_alltoall_combine"), f"{label} 不应有 alltoall"
        assert _find(ops, "moe_dispatch_post").forward(rt).op_subtype == "allreduce", (
            f"{label}: dispatch_post 必须解析为 allreduce"
        )
