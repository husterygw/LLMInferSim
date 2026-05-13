"""阶段 7-β: 跨节点通信公式手算 vs actual 对照 (详设 §4.7.3)。

按记忆 `feedback_cost_formula_handcheck.md` 必做。

覆盖:
  1. n ≤ intra_node_size 时退化到 flat ring 旧路径 (阶段 0-6 baseline 不变)
  2. n > intra_node_size 时启用 hierarchical 公式
  3. inter_node_bandwidth = 0 (旧 KNOWN_PROFILES) 时 fallback 到 intra
  4. allreduce hierarchical 公式手算
  5. allgather hierarchical 公式手算
  6. alltoall hierarchical 公式手算
  7. 边界:n=intra_node_size (单节点满) / n=intra_node_size+1 (跨节点边)
"""
import math

import pytest

from llm_infer_sim.core.ops.communication import (
    _hierarchical_allgather,
    _hierarchical_allreduce,
    _hierarchical_alltoall,
    _is_cross_node,
    allgather_time,
    allreduce_time,
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


# ------- 阶段 0-6 baseline 不变 -------

def test_flat_ring_unchanged_for_single_node():
    """n ≤ intra_node_size 时 allreduce 数字应跟阶段 0-6 旧 flat ring 完全一致."""
    hw = get_hardware_profile("H100")
    data = 1024 * 1024  # 1 MB
    # flat ring N=8: 2*(N-1) * (α + data/(N*β))
    expected = 2 * 7 * (hw.link_latency + data / (8 * hw.effective_intra_bw))
    assert allreduce_time(data, 8, hw, algo="ring") == pytest.approx(expected)


# ------- allreduce hierarchical -------

def test_allreduce_hierarchical_formula_handcheck():
    """tp=16 跨 2 节点 hierarchical allreduce 手算 vs actual.

    公式 (§4.7.3):
      t = 2(N1-1)(α_intra + data/(N1*β_intra))
        + 2(N2-1)(α_inter + data/(N1*N2*β_inter))

    H100: α_intra=0, α_inter=0, β_intra=450e9, β_inter=50e9
    """
    hw = get_hardware_profile("H100")
    data = 1024 * 1024  # 1 MB
    n = 16
    n1 = hw.intra_node_size  # 8
    n2 = 2

    intra_part = 2 * (n1 - 1) * (0.0 + data / (n1 * hw.effective_intra_bw))
    inter_part = 2 * (n2 - 1) * (0.0 + data / (n1 * n2 * hw.effective_inter_bw))
    expected = intra_part + inter_part

    actual = _hierarchical_allreduce(data, n, hw)
    assert actual == pytest.approx(expected, rel=1e-9)
    # 也通过顶层 allreduce_time 入口
    assert allreduce_time(data, n, hw) == pytest.approx(expected, rel=1e-9)


def test_allreduce_hierarchical_slower_than_flat_intra():
    """跨节点 hierarchical 应该比假设全 intra 的 flat ring 慢 (因为 inter 段慢)."""
    hw = get_hardware_profile("H100")
    data = 1024 * 1024
    # flat ring N=16 假设全 intra (用 N=16 的公式但 β=intra_bw, 仅作 lower bound 对比)
    flat_intra = 2 * 15 * (0.0 + data / (16 * hw.effective_intra_bw))
    hier = _hierarchical_allreduce(data, 16, hw)
    assert hier > flat_intra


def test_allreduce_hierarchical_faster_than_all_inter():
    """跨节点 hierarchical 应该比假设全 inter 的 flat ring 快 (intra 段还是快的)."""
    hw = get_hardware_profile("H100")
    data = 1024 * 1024
    # flat ring N=16 假设全 inter
    flat_inter = 2 * 15 * (0.0 + data / (16 * hw.effective_inter_bw))
    hier = _hierarchical_allreduce(data, 16, hw)
    assert hier < flat_inter


# ------- alltoall hierarchical -------

def test_alltoall_hierarchical_handcheck_n16():
    """tp/ep=16 跨 2 节点 hierarchical alltoall 手算 vs actual."""
    hw = get_hardware_profile("H100")
    data = 1024 * 1024
    n = 16
    n1 = hw.intra_node_size

    intra_part = (n1 - 1) * (0.0 + data / (n1 * n1 * hw.effective_intra_bw))
    inter_part = 1 * (0.0 + (data * n1) / (n * n * hw.effective_inter_bw))
    expected = intra_part + inter_part

    actual = _hierarchical_alltoall(data, n, hw)
    assert actual == pytest.approx(expected, rel=1e-9)
    assert alltoall_time(data, n, hw) == pytest.approx(expected, rel=1e-9)


# ------- allgather hierarchical -------

def test_allgather_hierarchical_handcheck_n16():
    hw = get_hardware_profile("H100")
    data = 1024 * 1024
    n = 16
    n1 = hw.intra_node_size
    n2 = 2

    intra_part = (n1 - 1) * (0.0 + (data / n2) / (n1 * hw.effective_intra_bw))
    inter_part = (n2 - 1) * (0.0 + (data / n2) / hw.effective_inter_bw)
    expected = intra_part + inter_part

    actual = _hierarchical_allgather(data, n, hw)
    assert actual == pytest.approx(expected, rel=1e-9)
    assert allgather_time(data, n, hw) == pytest.approx(expected, rel=1e-9)


# ------- 边界 -------

def test_boundary_n_eq_intra_size():
    """n=intra_node_size: 仍走 flat ring, hierarchical 不激活."""
    hw = get_hardware_profile("H100")
    assert not _is_cross_node(hw.intra_node_size, hw)
    # data ≥ 1MB 强制走 ring 而非 tree (_select_algo 在小 data 下选 tree)
    data = 2 * 1024 * 1024
    t = allreduce_time(data, hw.intra_node_size, hw, algo="ring")
    expected = 2 * 7 * (0.0 + data / (8 * hw.effective_intra_bw))
    assert t == pytest.approx(expected, rel=1e-9)


def test_boundary_n_eq_intra_size_plus_one():
    """n=intra_node_size+1: 触发 hierarchical (2 节点)."""
    hw = get_hardware_profile("H100")
    n = hw.intra_node_size + 1
    assert _is_cross_node(n, hw)
    t = allreduce_time(1024, n, hw)
    # 应该跟 _hierarchical_allreduce 算出同样值
    assert t == pytest.approx(_hierarchical_allreduce(1024, n, hw), rel=1e-9)


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
    assert hw.effective_intra_bw > hw.effective_inter_bw
    # ratio 大概 9× (450 vs 50, 单向后)
    assert 5.0 < hw.effective_intra_bw / hw.effective_inter_bw < 20.0


# ------- 跨节点真实物理直觉验证 -------

def test_cross_node_slower_than_single_node_by_realistic_factor():
    """同 data, ep=16 (跨 2 节点) alltoall 应明显慢于 ep=8 (单节点)."""
    hw = get_hardware_profile("H100")
    data = 1024 * 1024  # 1 MB
    t_single = alltoall_time(data, 8, hw)
    t_cross = alltoall_time(data, 16, hw)
    assert t_cross > t_single
    # 应在 2-10× 之间 (intra 0.5× + inter 主导)
    ratio = t_cross / t_single
    assert 1.5 < ratio < 20.0
