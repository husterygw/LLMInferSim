"""Phase 4 MoE + MoEDispatch static + forward contract equivalence.

Highest-risk migration after attention: the routed-experts formula (coupon-
collector distinct experts, grouped-GEMM flops/weight bytes) was pre-baked in the
constructor and is now recomputed from a step-resolved token count + a STATIC
routing skew. These tests lock byte-for-byte equivalence (spec / signature /
breakdown / backend latency) and that skew is honored as a static per-op param,
not a step input (op_plan §440)."""
from __future__ import annotations

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.graph.runtime import StepRuntime
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.operators.moe import (
    MoERoutingProfile, build_moe_dispatch, build_routed_experts,
)
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from tests.helpers.support import make_model_config

_T = 2048


def _ctx(tp=4, ep=4):
    return build_operator_context(
        make_model_config(hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
                    ffn_dim=6144, num_layers=48, vocab_size=151936,
                    is_moe=True, num_experts=128, num_activated_experts=8,
                    expert_dim=768, moe_layer_freq=1, first_moe_layer=0),
        DeploymentProfile.flat(tp=tp, ep=ep, moe_ep=ep),
        RuntimeProfile.flat(execution_mode="cudagraph",
                            backend="vllm", backend_version="0.19.1"),
        get_hardware_profile("RTX_4090"),
    )


def _routing():
    # per-layer skew so we exercise the skew-bucket path (layer 0 skew=0.3)
    return MoERoutingProfile(distribution="power_law", power_law_alpha=1.2,
                             skew=0.0, layer_skews=(0.3, 0.3, 0.0))


def _moe(ctx, tokens=_T, layer_idx=0):
    return build_routed_experts(ctx, _routing(), layer_idx, tokens=tokens,
                                phase="prefill")


def _step(t=_T):
    return StepRuntime(phase="prefill", total_tokens=t, num_prefill_tokens=t,
                       num_prefill_requests=1, execution_mode="cudagraph")


def test_moe_roofline_spec_equivalent():
    ctx = _ctx()
    op = _moe(ctx)
    assert op.roofline_spec(op.forward(_step())) == op.roofline_spec()


def test_moe_signature_equivalent():
    ctx = _ctx()
    op = _moe(ctx)
    rt = op.forward(_step())
    assert op.signature(rt) == op.signature()
    assert op.signature(rt).stable_hash() == op.signature().stable_hash()


def test_moe_backend_latency_equivalent():
    ctx = _ctx()
    op = _moe(ctx)
    rl = RooflineBackend(ctx.hw, ctx.execution_mode)
    assert rl.estimate(op, op.forward(_step())).latency_s == rl.estimate(op).latency_s


def test_moe_skew_is_static_not_step():
    """skew is folded at construction; forward only varies tokens. A layer with
    higher skew (fewer distinct experts → less weight read) must differ from a
    balanced layer at the SAME tokens."""
    ctx = _ctx()
    skewed = _moe(ctx, layer_idx=0)      # layer_skews[0] = 0.3
    balanced = _moe(ctx, layer_idx=2)    # layer_skews[2] = 0.0
    assert skewed.skew == 0.3 and balanced.skew == 0.0
    s_sk = skewed.roofline_spec(skewed.forward(_step()))
    s_ba = balanced.roofline_spec(balanced.forward(_step()))
    assert s_sk.load_weight != s_ba.load_weight  # skew changes distinct-expert weight read


def test_moe_forward_uses_step_tokens():
    ctx = _ctx()
    op = _moe(ctx)
    assert op.forward(_step(t=512)).shape["num_tokens"] == 512
    assert op.roofline_spec(op.forward(_step(512))) != op.roofline_spec(op.forward(_step(2048)))


# ---- MoEDispatch ----

def _dispatch(ctx, pre, tokens=_T):
    return build_moe_dispatch(ctx, pre_dispatch=pre, layer_idx=0, tokens=tokens,
                              phase="prefill")


def test_dispatch_pre_equivalent_to_legacy_local():
    """pre-dispatch 在 vLLM 永远本地 (无跨卡通信); roofline_spec recompute == baked."""
    ctx = _ctx()
    op = _dispatch(ctx, pre=True)
    assert op.roofline_spec(op.forward(_step())) == op.roofline_spec()


def test_dispatch_post_resolves_to_allreduce():
    """post-dispatch forward 按 tp/ep 解析为 allreduce(world=max(tp,ep))
    (AIC 对齐: MoEDispatch=通信, 折掉了独立 routed_expert_allreduce)."""
    ctx = _ctx()   # tp=4, ep=4 → max=4
    op = _dispatch(ctx, pre=False)
    rt = op.forward(_step())
    assert rt.op_subtype == "allreduce"
    assert rt.parallel["world_size"] == 4
    assert rt.shape["message_bytes"] == _T * ctx.model.hidden_dim * ctx.a_byte


def test_dispatch_forward_uses_step_tokens():
    ctx = _ctx()
    op = _dispatch(ctx, pre=True)
    assert op.forward(_step(512)).shape["num_tokens"] == 512
