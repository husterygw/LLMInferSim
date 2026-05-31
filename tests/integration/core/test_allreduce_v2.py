"""comm_plan Step 4 验收: AllReduce 走 hw.communication.backends["nccl"].allreduce.*
候选 ll_tree / ll128_tree / simple_ring / simple_tree 验证.

锁住 plan §10 Step 4 的 4 个数值验收点:
    n=2, 1KB:  选择 ll_tree, latency 约 7-15us
    n=4, 5KB:  选择 ll_tree 或 ll128_tree, latency < 25us
    n=8, 1KB:  latency ≈ 3 * ll_tree_alpha (21us, ±tolerance)
    n=4, 10MB: 选择 simple_ring 或 simple_tree (大消息), 不选 ll_tree

外加: RooflineBackend trace metadata 含 allreduce_candidates + selected_algorithm.
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.roofline.communication import (
    allreduce_time,
    allreduce_time_with_breakdown,
)
from llm_infer_sim.core.cost.router import CostRouter
from llm_infer_sim.core.step.step_plan import StepOpPlan
from llm_infer_sim.core.operators import Collective
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile


@pytest.fixture
def hw():
    return get_hardware_profile("RTX_4090")


# ---------------------------------------------------------------------------
# plan §10 Step 4 验收点 (走 allreduce_time_with_breakdown 直接拿候选)
# 全部 cudagraph 模式排除 framework_overhead 干扰.
# ---------------------------------------------------------------------------

def test_n2_1kb_picks_ll_tree(hw):
    """n=2, 1KB: 必选 ll_tree, latency 7-15us (depth=1 × 7us + tiny bw term)."""
    latency, br = allreduce_time_with_breakdown(
        data_bytes=1024, n=2, hw=hw, mode="cudagraph",
    )
    assert br["path"] == "nccl", f"未走 nccl: {br}"
    assert br["selected"] == "ll_tree", f"非 ll_tree: {br['selected']}"
    assert 7e-6 <= latency <= 15e-6, (
        f"n=2 1KB latency {latency*1e6:.1f}us 越界 [7,15]us"
    )


def test_n4_5kb_picks_ll_family(hw):
    """n=4, 5KB: 选 ll_tree 或 ll128_tree, latency < 40us."""
    latency, br = allreduce_time_with_breakdown(
        data_bytes=5 * 1024, n=4, hw=hw, mode="cudagraph",
    )
    assert br["path"] == "nccl"
    assert br["selected"] in ("ll_tree", "ll128_tree"), (
        f"非 LL family: {br['selected']}, candidates={br['candidates']}"
    )
    assert latency < 40e-6, f"n=4 5KB latency {latency*1e6:.1f}us 超 40us 上限"


def test_n8_1kb_scales_with_log2(hw):
    """n=8, 1KB: tree allreduce latency ≈ 2 * depth * ll_tree_alpha."""
    latency, br = allreduce_time_with_breakdown(
        data_bytes=1024, n=8, hw=hw, mode="cudagraph",
    )
    assert br["path"] == "nccl"
    assert br["selected"] == "ll_tree", f"n=8 1KB 应选 ll_tree, 实选 {br['selected']}"
    ll_tree_alpha = hw.communication.backends["nccl"].allreduce.ll_tree_alpha_s
    expected = 2 * 3 * ll_tree_alpha
    assert 0.8 * expected <= latency <= 1.5 * expected, (
        f"n=8 1KB latency {latency*1e6:.1f}us 偏离 2*depth*alpha={expected*1e6:.1f}us"
    )


def test_n4_10mb_picks_bw_path(hw):
    """n=4, 10MB: 大消息走 simple_ring / simple_tree, 不选 ll_tree."""
    latency, br = allreduce_time_with_breakdown(
        data_bytes=10 * 1024 * 1024, n=4, hw=hw, mode="cudagraph",
    )
    assert br["path"] == "nccl"
    assert br["selected"] in ("simple_ring", "simple_tree"), (
        f"10MB 仍选 {br['selected']}: candidates={br['candidates']}"
    )
    # 10MB > ll_tree_max_bytes 也 > ll128_tree_max_bytes → ll_* 不入候选
    assert "ll_tree" not in br["candidates"], br["candidates"]
    assert "ll128_tree" not in br["candidates"], br["candidates"]
    assert latency > 0


# ---------------------------------------------------------------------------
# breakdown 输出 + RooflineBackend metadata 形态
# ---------------------------------------------------------------------------

def test_breakdown_contains_all_candidates(hw):
    """5KB n=4 → 4 个候选 (ll_tree, ll128_tree, simple_ring, simple_tree)."""
    _, br = allreduce_time_with_breakdown(
        data_bytes=5 * 1024, n=4, hw=hw, mode="cudagraph",
    )
    assert set(br["candidates"]) == {
        "ll_tree", "ll128_tree", "simple_ring", "simple_tree",
    }


def test_roofline_metadata_carries_breakdown(hw):
    """走 RooflineBackend 全链路, metadata 应出现 allreduce_candidates / selected_algorithm."""
    from llm_infer_sim.core.operators.context import build_operator_context
    from tests.helpers.support import make_model_config

    deployment = DeploymentProfile.flat(tp=4)
    runtime = RuntimeProfile.flat(
        backend="vllm", backend_version="0.20.1",
        execution_mode="cudagraph",
    )
    ctx = build_operator_context(make_model_config(), deployment, runtime, hw)
    op = Collective(
        name="attn_ar", op_subtype="allreduce",
        phase="decode", layer_idx=0,
        message_bytes=5 * 1024, world_size=4,
        ctx=ctx, comm_backend="nccl",
        roofline_spec_value=RooflineSpec(
            comm_bytes=5 * 1024, comm_type="allreduce",
            op_category="communication",
        ),
    )
    be = RooflineBackend(hw, "cudagraph")
    router = CostRouter(be)
    trace = router.estimate(StepOpPlan(step_id=0, phase="decode", ops=(op,)))
    md = trace.entries[0].metadata
    assert md["communication_path"] == "nccl"
    assert md["selected_algorithm"] in ("ll_tree", "ll128_tree")
    assert "allreduce_candidates" in md
    assert "ll_tree" in md["allreduce_candidates"]
    assert "simple_ring" in md["allreduce_candidates"]
    assert md["algorithm_term_s"] > 0


# ---------------------------------------------------------------------------
# algo / protocol hint
# ---------------------------------------------------------------------------

def test_algo_hint_forces_candidate(hw):
    """algo='simple_ring' 时强制选 ring 即使 ll_tree 更快."""
    _, br = allreduce_time_with_breakdown(
        data_bytes=1024, n=4, hw=hw, mode="cudagraph", algo="simple_ring",
    )
    assert br["selected"] == "simple_ring"


def test_protocol_hint_filters_candidates(hw):
    """protocol_hint='simple' 时只剩 simple_ring / simple_tree."""
    _, br = allreduce_time_with_breakdown(
        data_bytes=1024, n=4, hw=hw, mode="cudagraph", protocol_hint="simple",
    )
    assert set(br["candidates"]) <= {"simple_ring", "simple_tree"}
    assert br["selected"] in ("simple_ring", "simple_tree")


# ---------------------------------------------------------------------------
# wrapper scalar API 兼容
# ---------------------------------------------------------------------------

def test_allreduce_time_returns_scalar(hw):
    """老 allreduce_time(...) 入口仍返 float (跟 with_breakdown 的 latency 一致)."""
    latency, _ = allreduce_time_with_breakdown(1024, 4, hw, mode="cudagraph")
    assert allreduce_time(1024, 4, hw, mode="cudagraph") == pytest.approx(latency)
