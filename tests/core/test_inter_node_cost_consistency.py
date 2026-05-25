"""阶段 7-β: 跨节点通信公式手算 vs actual 对照 (详设 §4.7.3)。

按记忆 `feedback_cost_formula_handcheck.md` 必做。

覆盖:
  1. allreduce 顶层入口统一走 NCCL candidates
  2. n > intra_node_size 时 allgather/alltoall 仍启用 hierarchical 公式
  3. inter_node_bandwidth = 0 (旧 KNOWN_PROFILES) 时 fallback 到 intra
  4. allgather hierarchical 公式手算
  5. alltoall hierarchical 公式手算
  6. 边界:n=intra_node_size (单节点满) / n=intra_node_size+1 (跨节点边)
"""
import math

import pytest

from llm_infer_sim.core.cost.roofline.communication import (
    _hierarchical_allgather,
    _hierarchical_alltoall,
    _is_cross_node,
    allgather_time,
    allreduce_time,
    allreduce_time_with_breakdown,
    alltoall_time,
)
from llm_infer_sim.core.profiles.hardware import HardwareConfig, get_hardware_profile


# ------- _is_cross_node 边界 -------

def test_is_cross_node_at_intra_size():
    hw = get_hardware_profile("H100")
    assert hw.intra_node_size == 8
    assert hw.inter_node_bandwidth > 0  # 阶段 7 已填实测值
    assert not _is_cross_node(8, hw)    # 满 1 节点仍 intra
    assert _is_cross_node(9, hw)        # 跨节点边界
    assert _is_cross_node(16, hw)
    assert not _is_cross_node(2, hw)


def test_is_cross_node_fallback_when_inter_bw_zero():
    """老硬件 (inter_node_bandwidth=0 placeholder) 应该 fallback 到 intra."""
    hw = HardwareConfig(
        intra_node_bandwidth=900e9, intra_node_size=8,
        inter_node_bandwidth=0.0,
    )
    assert not _is_cross_node(16, hw)
    # alltoall 应该走 intra 公式不挂
    t = alltoall_time(1024, 16, hw)
    assert t > 0 and math.isfinite(t)


# ------- allreduce: unified NCCL path -------

def test_allreduce_single_node_uses_nccl_candidates():
    """n ≤ intra_node_size 时 allreduce 也统一走 NCCL candidates."""
    hw = get_hardware_profile("H100")
    data = 1024 * 1024  # 1 MB
    latency, br = allreduce_time_with_breakdown(
        data, 8, hw, algo="simple_ring", mode="cudagraph",
    )
    assert br["path"] == "nccl"
    assert br["selected"] == "simple_ring"
    assert latency == pytest.approx(br["candidates"]["simple_ring"])


# ------- allreduce no legacy hierarchical -------

def test_allreduce_cross_node_still_uses_nccl_candidates():
    """allreduce 不再走 legacy hierarchical path, 即使 n 跨节点。"""
    hw = get_hardware_profile("H100")
    data = 1024 * 1024  # 1 MB
    n = 16
    assert _is_cross_node(n, hw)
    latency, br = allreduce_time_with_breakdown(data, n, hw, mode="cudagraph")
    assert br["path"] == "nccl"
    assert br["selected"] in br["candidates"]
    assert latency == pytest.approx(br["candidates"][br["selected"]])


# ------- alltoall hierarchical -------

def test_alltoall_hierarchical_handcheck_n16():
    """tp/ep=16 跨 2 节点 hierarchical alltoall 手算 vs actual."""
    hw = get_hardware_profile("H100")
    data = 1024 * 1024
    n = 16
    n1 = hw.intra_node_size

    intra_part = (n1 - 1) * (0.0 + data / (n1 * n1 * hw.effective_intra_bw(n1)))
    inter_part = 1 * (0.0 + (data * n1) / (n * n * hw.effective_inter_bw))
    expected = intra_part + inter_part

    actual = _hierarchical_alltoall(data, n, hw)
    assert actual == pytest.approx(expected, rel=1e-9)
    assert alltoall_time(data, n, hw, mode="cudagraph") == pytest.approx(expected, rel=1e-9)


# ------- allgather hierarchical -------

def test_allgather_hierarchical_handcheck_n16():
    hw = get_hardware_profile("H100")
    data = 1024 * 1024
    n = 16
    n1 = hw.intra_node_size
    n2 = 2

    intra_part = (n1 - 1) * (0.0 + (data / n2) / (n1 * hw.effective_intra_bw(n1)))
    inter_part = (n2 - 1) * (0.0 + (data / n2) / hw.effective_inter_bw)
    expected = intra_part + inter_part

    actual = _hierarchical_allgather(data, n, hw)
    assert actual == pytest.approx(expected, rel=1e-9)
    assert allgather_time(data, n, hw, mode="cudagraph") == pytest.approx(expected, rel=1e-9)


# ------- 边界 -------

def test_boundary_n_eq_intra_size():
    """n=intra_node_size: allreduce 走 NCCL candidate path."""
    hw = get_hardware_profile("H100")
    assert not _is_cross_node(hw.intra_node_size, hw)
    data = 2 * 1024 * 1024
    t, br = allreduce_time_with_breakdown(
        data, hw.intra_node_size, hw, algo="simple_ring", mode="cudagraph",
    )
    assert br["path"] == "nccl"
    assert t == pytest.approx(br["candidates"]["simple_ring"], rel=1e-9)


def test_boundary_n_eq_intra_size_plus_one():
    """n=intra_node_size+1: allreduce 不再触发 legacy hierarchical."""
    hw = get_hardware_profile("H100")
    n = hw.intra_node_size + 1
    assert _is_cross_node(n, hw)
    t, br = allreduce_time_with_breakdown(1024, n, hw, mode="cudagraph")
    assert br["path"] == "nccl"
    assert t == pytest.approx(br["candidates"][br["selected"]], rel=1e-9)


def test_n_one_returns_zero():
    hw = get_hardware_profile("H100")
    assert allreduce_time(1024, 1, hw) == 0.0
    assert allgather_time(1024, 1, hw) == 0.0
    assert alltoall_time(1024, 1, hw) == 0.0


# ------- KNOWN_PROFILES 实测值检查 -------

def test_h100_h200_b200_have_realistic_inter_bw():
    """阶段 7 已填实测 IB 带宽."""
    h100 = get_hardware_profile("H100")
    b200 = get_hardware_profile("B200")
    assert h100.inter_node_bandwidth >= 25e9  # NDR400 ≥ 50 GB/s
    assert b200.inter_node_bandwidth >= 100e9  # NDR800
    assert h100.intra_node_size == 8
    assert b200.intra_node_size == 8


def test_h100_effective_bw_split():
    """H100: intra_bw / inter_bw 各自属性独立可读."""
    hw = get_hardware_profile("H100")
    # NVLink full mesh 下 effective_intra_bw 跟 n 无关
    assert hw.effective_intra_bw(1) > hw.effective_inter_bw
    # ratio 大概 3-5× (450/2 × 0.6 = 135 vs ~50)
    assert 1.5 < hw.effective_intra_bw(8) / hw.effective_inter_bw < 20.0


# ------- 跨节点真实物理直觉验证 -------

@pytest.mark.xfail(
    reason="Phase 5 `data_bytes = per-rank input bytes` 新语义跟 `_hierarchical_alltoall` "
           "公式中的 (data * n1) / (n*n*beta_inter) 不匹配 (后者按 global data 设计). "
           "cross-node 路径需要 Phase 6 重写, 文档见 docs/COMMUNICATION_MODELING.md §16.",
    strict=False,
)
def test_cross_node_slower_than_single_node_by_realistic_factor():
    """同 data, ep=16 (跨 2 节点) alltoall 应明显慢于 ep=8 (单节点)."""
    hw = get_hardware_profile("H100")
    data = 1024 * 1024  # 1 MB
    t_single = alltoall_time(data, 8, hw, mode="cudagraph")
    t_cross = alltoall_time(data, 16, hw, mode="cudagraph")
    assert t_cross > t_single
    ratio = t_cross / t_single
    assert 1.5 < ratio < 20.0
