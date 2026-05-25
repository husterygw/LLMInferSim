"""comm_plan Step 6-A 验收: AllToAll 走 hw.communication.backends["nccl"].alltoall.*
候选 pairwise + contention_factor (n>4) 验证.

跟 test_allreduce_v2.py 对称, 锁住:
  - hw.communication 配置时 path == "v2"
  - n=4 candidates 含 "pairwise", n=8 应用 contention_factor 放大 ~1.2x
  - RooflineBackend trace metadata 含 alltoall_candidates + selected_algorithm
  - algo_hint 强制候选
  - scalar wrapper 兼容
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.roofline.communication import (
    alltoall_time,
    alltoall_time_with_breakdown,
)
from llm_infer_sim.core.cost.router import CostRouter
from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.operators import AllToAll
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile


@pytest.fixture
def hw():
    return get_hardware_profile("RTX_4090")


# ---------------------------------------------------------------------------
# v2 path basic invariants
# ---------------------------------------------------------------------------

def test_v2_path_when_communication_configured(hw):
    """RTX_4090 hw.communication 已配 → AllToAll 走 v2."""
    _, br = alltoall_time_with_breakdown(
        data_bytes=1024 * 1024, n=4, hw=hw, mode="cudagraph",
    )
    assert br["path"] == "v2"
    assert br["selected"] == "pairwise"
    assert "pairwise" in br["candidates"]


def test_legacy_fallback_when_no_communication(hw):
    """hw.communication=None 时退化到 legacy_intra 路径."""
    import dataclasses
    hw_legacy = dataclasses.replace(hw, communication=None)
    _, br = alltoall_time_with_breakdown(
        data_bytes=1024 * 1024, n=4, hw=hw_legacy, mode="cudagraph",
    )
    assert br["path"] == "legacy_intra"
    assert br["selected"] == "pairwise"


# ---------------------------------------------------------------------------
# contention_factor: n>4 时放大
# ---------------------------------------------------------------------------

def test_contention_factor_applies_for_n_gt_4(hw):
    """n=8 pairwise 相对 n=4 同 data 应放大 contention_factor (默认 1.2).

    比较等长 message: 8-rank pairwise / 4-rank pairwise:
      base ratio = (n-1)/n × bytes/beta / 4-base 主要 BW 项, 加上 (n-1) alpha 占小.
      contention 1.2 lift 在 8-rank.
    锁住 sim 在 n=4 -> n=8 时正确放大 contention.
    """
    # 直接拉 candidates 对比 (排除 framework overhead)
    _, br4 = alltoall_time_with_breakdown(64 * 1024, 4, hw, mode="cudagraph")
    _, br8 = alltoall_time_with_breakdown(64 * 1024, 8, hw, mode="cudagraph")
    p4 = br4["candidates"]["pairwise"]
    p8 = br8["candidates"]["pairwise"]
    params = hw.communication.backends["nccl"].alltoall
    cf = params.contention_factor
    # n=8 的 pairwise 必须含 contention 因子
    # n=8 不含 contention 的"hypothetical" 时长大致 ≈ p4 × 7/3 (近似), 但 BW 项主导.
    # 这里只验证 contention 实际生效: 测一个 dummy 不带 contention 的算法 (从公式手算)
    # 等价于把 contention 拿掉的 ratio = 1 / cf.
    ratio = p8 / (p8 / cf)   # tautology, just to express intent
    assert ratio == pytest.approx(cf)  # contention applied
    # 强一点的: 把 hw 拷一份 contention=1.0 比较
    import dataclasses
    new_alltoall = dataclasses.replace(params, contention_factor=1.0)
    new_backend = dataclasses.replace(
        hw.communication.backends["nccl"], alltoall=new_alltoall,
    )
    new_comm = dataclasses.replace(
        hw.communication, backends={"nccl": new_backend},
    )
    hw_no_contention = dataclasses.replace(hw, communication=new_comm)
    _, br8_no = alltoall_time_with_breakdown(64 * 1024, 8, hw_no_contention, mode="cudagraph")
    p8_no = br8_no["candidates"]["pairwise"]
    assert p8 == pytest.approx(p8_no * cf, rel=1e-9), (
        f"contention not applied properly: p8={p8*1e6:.2f}us p8_no={p8_no*1e6:.2f}us cf={cf}"
    )


def test_contention_not_applied_at_n4(hw):
    """n=4 (≤4) contention_factor 不放大 (注释: 'n>4 时' )."""
    import dataclasses
    _, br = alltoall_time_with_breakdown(64 * 1024, 4, hw, mode="cudagraph")
    p4 = br["candidates"]["pairwise"]

    new_alltoall = dataclasses.replace(
        hw.communication.backends["nccl"].alltoall, contention_factor=1.0,
    )
    new_backend = dataclasses.replace(
        hw.communication.backends["nccl"], alltoall=new_alltoall,
    )
    new_comm = dataclasses.replace(
        hw.communication, backends={"nccl": new_backend},
    )
    hw_no_contention = dataclasses.replace(hw, communication=new_comm)
    _, br_no = alltoall_time_with_breakdown(64 * 1024, 4, hw_no_contention, mode="cudagraph")
    p4_no = br_no["candidates"]["pairwise"]
    assert p4 == pytest.approx(p4_no, rel=1e-9), (
        f"contention shouldn't apply at n=4: p4={p4*1e6:.2f}us p4_no={p4_no*1e6:.2f}us"
    )


# ---------------------------------------------------------------------------
# RooflineBackend metadata 含 alltoall_candidates
# ---------------------------------------------------------------------------

def test_roofline_metadata_carries_alltoall_breakdown(hw):
    """RooflineBackend metadata 含 alltoall_candidates / selected_algorithm."""
    from llm_infer_sim.core.operators.context import build_operator_context
    from llm_infer_sim.core.profiles.model_config import ModelConfig

    deploy = DeployConfig(
        tp_size=4, backend="vllm", backend_version="0.20.1",
        execution_mode="cudagraph",
    )
    ctx = build_operator_context(ModelConfig(), deploy, hw)
    op = AllToAll(
        name="ep_alltoall_dispatch", op_subtype="alltoall",
        phase="decode", layer_idx=0,
        message_bytes=64 * 1024, world_size=4,
        ctx=ctx, comm_backend="nccl",
        roofline_spec_value=RooflineSpec(
            comm_bytes=64 * 1024, comm_type="alltoall",
            op_category="communication",
        ),
    )
    be = RooflineBackend(hw, deploy)
    router = CostRouter(be)
    trace = router.estimate(StepOpPlan(step_id=0, phase="decode", ops=(op,)))
    md = trace.entries[0].metadata
    assert md["communication_path"] == "v2"
    assert md["selected_algorithm"] == "pairwise"
    assert "alltoall_candidates" in md
    assert "pairwise" in md["alltoall_candidates"]
    assert md["algorithm_term_s"] > 0


# ---------------------------------------------------------------------------
# algo / scalar wrapper
# ---------------------------------------------------------------------------

def test_algo_hint_forces_candidate(hw):
    """algo='pairwise' 单候选时跟 auto 等价."""
    _, br = alltoall_time_with_breakdown(
        1024, 4, hw, mode="cudagraph", algo="pairwise",
    )
    assert br["selected"] == "pairwise"


def test_alltoall_time_returns_scalar(hw):
    """老 alltoall_time(...) 入口仍返 float."""
    latency, _ = alltoall_time_with_breakdown(1024, 4, hw, mode="cudagraph")
    assert alltoall_time(1024, 4, hw, mode="cudagraph") == pytest.approx(latency)


def test_short_circuit_n1(hw):
    """n=1 / data=0 → 0.0, breakdown 有 path."""
    t, br = alltoall_time_with_breakdown(1024, 1, hw, mode="cudagraph")
    assert t == 0.0
    assert br["selected"] is None
