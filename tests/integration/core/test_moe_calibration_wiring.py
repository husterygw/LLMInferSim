"""moe_plan §5.A 验收: MoE calibration knobs 必须真的影响 trace, 非 dead wiring.

锁住 plan §5.A step 4:
  - 修改 topk_overhead_us 只改变 moe_topk op latency / t_topk metadata
  - 修改 local_dispatch_overhead_us 只改变 MoEDispatch op latency / t_dispatch_local
  - 修改 grouped_gemm_efficiency 只改变 MoE op (routed_experts) latency / t_expert_compute
  - moe_profile_id 出现在受影响 op metadata 上, 用于 trace 调试
  - 其它 op (GEMM / Norm / Attention / Collective) 不受 MoE knob 影响

依据 project_efficiency_profile_dead_wiring 教训: 必须证非 dead, 不能只看 dataclass 字段值.
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.router import CostRouter
from llm_infer_sim.core.step.step_plan import StepOpPlan
from llm_infer_sim.core.operators import (
    AllReduce, ElementWise, GEMM, MoERoutingProfile,
    build_moe_dispatch, build_routed_experts,
)
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from tests.helpers.support import make_model_config
from llm_infer_sim.core.calibration import CalibrationProfile, MoEEfficiencyProfile


def _qwen_moe_ctx(tp: int = 4, ep: int = 1):
    mc = make_model_config(
        name="Qwen3-30B-A3B",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=0, num_layers=48, vocab_size=151936,
        is_moe=True, num_experts=128, num_activated_experts=8, expert_dim=768,
    )
    deployment = DeploymentProfile.flat(tp=tp, ep=ep)
    runtime = RuntimeProfile.flat(
        backend="vllm", backend_version="0.20.1",
        execution_mode="cudagraph",
    )
    ctx = build_operator_context(mc, deployment, runtime, get_hardware_profile("RTX_4090"))
    return mc, "cudagraph", ctx


def _hw_with_profile(profile: MoEEfficiencyProfile) -> CalibrationProfile:
    # Step G: MoE calibration knob 现挂 CalibrationProfile, 不再 masquerade 成 hw spec.
    return CalibrationProfile(moe_efficiency=profile)


def _estimate(op, calibration, deploy):
    hw = get_hardware_profile("RTX_4090")
    # deploy is the execution_mode string
    backend = RooflineBackend(hw, deploy, calibration=calibration)
    router = CostRouter(backend)
    trace = router.estimate(StepOpPlan(step_id=0, phase="decode", ops=(op,)))
    return trace.entries[0]


# ---------------------------------------------------------------------------
# topk_overhead_us
# ---------------------------------------------------------------------------

def test_topk_overhead_changes_only_topk_latency():
    """改 topk_overhead_us 必须 (a) 改 moe_topk latency, (b) t_topk metadata, (c) 不影响其它 op."""
    mc, deploy, ctx = _qwen_moe_ctx()
    topk_op = ElementWise(
        name="moe_topk", op_subtype="topk",
        phase="decode", layer_idx=0,
        tokens=128, intermediate=mc.num_experts, ctx=ctx,
    )

    hw_zero = _hw_with_profile(MoEEfficiencyProfile(profile_id="zero"))
    hw_calib = _hw_with_profile(MoEEfficiencyProfile(
        profile_id="topk_5us", topk_overhead_us=5.0,
    ))

    e_zero = _estimate(topk_op, hw_zero, deploy)
    e_calib = _estimate(topk_op, hw_calib, deploy)

    # latency 差正好 5us
    delta_s = e_calib.latency_s - e_zero.latency_s
    assert delta_s == pytest.approx(5e-6, abs=1e-9), (
        f"topk_overhead_us=5 应使 latency 增 5us, 实际 +{delta_s*1e6:.2f}us"
    )
    # metadata 写 t_topk + moe_profile_id
    assert e_calib.metadata.get("t_topk") == e_calib.latency_s
    assert e_calib.metadata.get("topk_overhead_us") == 5.0
    assert e_calib.metadata.get("moe_profile_id") == "topk_5us"
    # zero profile 仍有 metadata key (overhead=0 也算 applied)
    assert "t_topk" in e_zero.metadata


def test_topk_overhead_does_not_affect_gemm():
    """改 topk_overhead_us 不能影响 GEMM op (隔离验证)."""
    mc, deploy, ctx = _qwen_moe_ctx()
    gemm_op = GEMM(
        name="qkv_proj", op_subtype="qkv_proj",
        phase="decode", layer_idx=0,
        m=128, n=6144, k=2048, ctx=ctx,
    )

    hw_zero = _hw_with_profile(MoEEfficiencyProfile(profile_id="zero"))
    hw_calib = _hw_with_profile(MoEEfficiencyProfile(
        profile_id="topk_99us", topk_overhead_us=99.0,
    ))

    e_zero = _estimate(gemm_op, hw_zero, deploy)
    e_calib = _estimate(gemm_op, hw_calib, deploy)
    assert e_zero.latency_s == pytest.approx(e_calib.latency_s)
    # GEMM 不写 MoE metadata
    assert "t_topk" not in e_calib.metadata
    assert "moe_profile_id" not in e_calib.metadata


# ---------------------------------------------------------------------------
# local_dispatch_overhead_us
# ---------------------------------------------------------------------------

def test_local_dispatch_overhead_changes_only_dispatch_latency():
    """改 local_dispatch_overhead_us 仅影响 MoEDispatch op."""
    mc, deploy, ctx = _qwen_moe_ctx()
    md_op = build_moe_dispatch(
        ctx, pre_dispatch=True, layer_idx=0, tokens=128, phase="decode",
    )

    hw_zero = _hw_with_profile(MoEEfficiencyProfile(profile_id="zero"))
    hw_calib = _hw_with_profile(MoEEfficiencyProfile(
        profile_id="dispatch_3us", local_dispatch_overhead_us=3.0,
    ))

    e_zero = _estimate(md_op, hw_zero, deploy)
    e_calib = _estimate(md_op, hw_calib, deploy)
    delta_s = e_calib.latency_s - e_zero.latency_s
    assert delta_s == pytest.approx(3e-6, abs=1e-9), (
        f"local_dispatch_overhead_us=3 应使 latency 增 3us, 实际 +{delta_s*1e6:.2f}us"
    )
    assert e_calib.metadata.get("t_dispatch_local") == e_calib.latency_s
    assert e_calib.metadata.get("local_dispatch_overhead_us") == 3.0
    assert e_calib.metadata.get("moe_profile_id") == "dispatch_3us"


def test_local_dispatch_overhead_does_not_affect_topk():
    """改 local_dispatch_overhead_us 不影响 moe_topk."""
    mc, deploy, ctx = _qwen_moe_ctx()
    topk_op = ElementWise(
        name="moe_topk", op_subtype="topk",
        phase="decode", layer_idx=0,
        tokens=128, intermediate=mc.num_experts, ctx=ctx,
    )
    hw_zero = _hw_with_profile(MoEEfficiencyProfile(profile_id="zero"))
    hw_calib = _hw_with_profile(MoEEfficiencyProfile(
        profile_id="dispatch_only", local_dispatch_overhead_us=99.0,
    ))
    e_zero = _estimate(topk_op, hw_zero, deploy)
    e_calib = _estimate(topk_op, hw_calib, deploy)
    assert e_zero.latency_s == pytest.approx(e_calib.latency_s)


# ---------------------------------------------------------------------------
# grouped_gemm_efficiency
# ---------------------------------------------------------------------------

def test_grouped_gemm_efficiency_scales_only_moe_latency():
    """改 grouped_gemm_efficiency 仅影响 MoE (routed_experts) op latency.

    new_latency = roofline_latency / efficiency. eff=0.5 → 2× latency.
    """
    mc, deploy, ctx = _qwen_moe_ctx()
    moe_op = build_routed_experts(
        ctx, MoERoutingProfile.balanced(), 0, tokens=128, phase="prefill",
    )
    hw_baseline = _hw_with_profile(MoEEfficiencyProfile(
        profile_id="eff_1", grouped_gemm_efficiency=1.0,
    ))
    hw_half = _hw_with_profile(MoEEfficiencyProfile(
        profile_id="eff_0.5", grouped_gemm_efficiency=0.5,
    ))

    e_base = _estimate(moe_op, hw_baseline, deploy)
    e_half = _estimate(moe_op, hw_half, deploy)
    # 0.5 efficiency → latency 应翻倍
    assert e_half.latency_s == pytest.approx(e_base.latency_s * 2.0, rel=1e-9)
    # metadata
    assert e_half.metadata.get("grouped_gemm_efficiency") == 0.5
    assert e_half.metadata.get("t_expert_compute") == e_half.latency_s
    assert e_half.metadata.get("moe_profile_id") == "eff_0.5"
    # roofline_s 保持不变 (反映 pre-calibration latency)
    assert e_half.roofline_s == pytest.approx(e_base.roofline_s, rel=1e-9)


def test_grouped_gemm_efficiency_does_not_affect_topk_or_dispatch():
    """grouped_gemm_efficiency 隔离: 不影响 moe_topk / MoEDispatch."""
    mc, deploy, ctx = _qwen_moe_ctx()
    topk_op = ElementWise(
        name="moe_topk", op_subtype="topk",
        phase="decode", layer_idx=0,
        tokens=128, intermediate=mc.num_experts, ctx=ctx,
    )
    md_op = build_moe_dispatch(
        ctx, pre_dispatch=True, layer_idx=0, tokens=128, phase="decode",
    )

    hw_baseline = _hw_with_profile(MoEEfficiencyProfile(profile_id="b"))
    hw_eff = _hw_with_profile(MoEEfficiencyProfile(
        profile_id="eff_only", grouped_gemm_efficiency=0.3,
    ))

    for op in (topk_op, md_op):
        e_b = _estimate(op, hw_baseline, deploy)
        e_e = _estimate(op, hw_eff, deploy)
        assert e_b.latency_s == pytest.approx(e_e.latency_s), (
            f"grouped_gemm_efficiency should NOT affect {op.op_kind}/{op.op_subtype}"
        )


# ---------------------------------------------------------------------------
# moe_efficiency=None 退化 (向后兼容)
# ---------------------------------------------------------------------------

def test_moe_efficiency_none_skips_calibration():
    """calibration.moe_efficiency=None 时 latency 跟 roofline 一致, 不写 MoE metadata 字段."""
    mc, deploy, ctx = _qwen_moe_ctx()
    moe_op = build_routed_experts(
        ctx, MoERoutingProfile.balanced(), 0, tokens=128, phase="prefill",
    )
    calib_none = CalibrationProfile(moe_efficiency=None)
    e = _estimate(moe_op, calib_none, deploy)
    assert e.latency_s == e.roofline_s
    assert "t_expert_compute" not in e.metadata
    assert "grouped_gemm_efficiency" not in e.metadata
    assert "moe_profile_id" not in e.metadata


# ---------------------------------------------------------------------------
# AllReduce (collective) 不受影响
# ---------------------------------------------------------------------------

def test_collective_op_not_affected_by_moe_calibration():
    """collective op 走 _estimate_collective 独立 path, MoE knob 不应影响."""
    mc, deploy, ctx = _qwen_moe_ctx()
    ar = AllReduce(
        name="ar", op_subtype="allreduce",
        phase="decode", layer_idx=0,
        message_bytes=5 * 1024, world_size=4, ctx=ctx,
        roofline_spec_value=RooflineSpec(
            comm_bytes=5 * 1024, comm_type="allreduce",
            op_category="communication",
        ),
    )
    hw_zero = _hw_with_profile(MoEEfficiencyProfile(profile_id="z"))
    hw_calib = _hw_with_profile(MoEEfficiencyProfile(
        profile_id="all", topk_overhead_us=99.0,
        local_dispatch_overhead_us=99.0, grouped_gemm_efficiency=0.1,
    ))
    e_zero = _estimate(ar, hw_zero, deploy)
    e_calib = _estimate(ar, hw_calib, deploy)
    assert e_zero.latency_s == pytest.approx(e_calib.latency_s)
    assert "moe_profile_id" not in e_calib.metadata
