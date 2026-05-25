"""Hardware profile registry + RTX_4090 (详设 §9.4.2 Plan B / B.0).

覆盖:
  1. RTX_4090 profile 注册 + 关键字段合理性 (BF16 ~165 TFLOPS, BW ~1 TB/s, 24GB)
  2. RTX_4090 别名 (rtx_4090, RTX4090, nvidia_RTX_4090) 都能解析到同一 profile
  3. RTX_4090 ridge point 在合理范围 (~160 FLOP/byte 对 BF16)
  4. RTX_4090 没 NVLink → intra_node_bandwidth 远小于 H100 (32 vs 600 GB/s)
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.profiles.hardware import (
    KNOWN_PROFILES,
    PROFILE_ALIASES,
    get_hardware_profile,
)


def test_rtx_4090_in_known_profiles():
    assert "RTX_4090" in KNOWN_PROFILES


def test_rtx_4090_specs_match_ada_whitepaper():
    """根据 NVIDIA Ada whitepaper + 4090 page 校验主要数字."""
    hw = get_hardware_profile("RTX_4090")
    # BF16/FP16 dense TC: 165.2 TFLOPS
    assert hw.peak_flops_fp16 == pytest.approx(165.2e12, rel=0.01)
    assert hw.peak_flops_bf16 == pytest.approx(165.2e12, rel=0.01)
    # FP8 E4M3 dense TC: 2× BF16 = ~660 TFLOPS
    assert hw.peak_flops_fp8 == pytest.approx(660.6e12, rel=0.01)
    # FP4 在 Ada 不支持
    assert hw.peak_flops_fp4 == 0.0
    assert hw.has_fp4_tc is False
    # FP32 CUDA core
    assert hw.vector_flops == pytest.approx(82.6e12, rel=0.01)
    # GDDR6X bandwidth ~1 TB/s
    assert hw.mem_bandwidth == pytest.approx(1008e9, rel=0.01)
    # 24 GB
    assert hw.mem_capacity_gb == 24
    # On-chip = register file aggregate (跟 A100/H100 一个口径),
    # AD102 128 SM × 256 KB/SM = 32 MB. 不是 L2 (L2 是 72 MB,
    # 跨 kernel 共享, 跟 FA tile sizing 无关).
    assert hw.onchip_buffer == pytest.approx(32 * 1000 * 1000, rel=0.05)


def test_rtx_4090_no_nvlink_pcie_only():
    """RTX 4090 消费卡无 NVLink, intra_node 走 PCIe 4.0 ×16: 32 GB/s 单向 / 64 GB/s 双向 nominal.
    Phase 5: 拓扑感知, n_per_root concentrated 默认对应 same-NUMA.
    2026-05-17 baseline: protocol_efficiency=0.625 (独立实测 measure_allreduce.py)."""
    hw = get_hardware_profile("RTX_4090")
    assert hw.intra_node_bandwidth == 64e9   # PCIe 4.0 ×16 bidirectional nominal
    assert hw.intra_node_topology == "pcie_shared_root"
    assert hw.intra_node_gpus_per_root == 4
    assert hw.intra_node_num_roots == 2
    assert hw.inter_node_bandwidth == 0.0
    # n-缩放: β = (64/2) × 0.625 / n_per_root
    # TP=4 same-NUMA (n_per_root=4): β = 32 × 0.625 / 4 = 5.0 GB/s ← 实测校准点
    assert abs(hw.effective_intra_bw(4, "concentrated") - 5e9) < 0.1e9
    # TP=2 same-NUMA (n_per_root=2): β = 32 × 0.625 / 2 = 10 GB/s
    assert abs(hw.effective_intra_bw(2, "concentrated") - 10e9) < 0.1e9
    # TP=2 balanced (n_per_root = ceil(2/2) = 1): β = 32 × 0.625 = 20 GB/s
    assert abs(hw.effective_intra_bw(2, "balanced") - 20e9) < 0.1e9
    # visible_devices: GPU 0,4 = cross-NUMA → n_per_root=1 → β=20
    assert abs(hw.effective_intra_bw(2, visible_devices=[0, 4]) - 20e9) < 0.1e9


def test_rtx_4090_aliases_resolve():
    """所有 alias 都映射到同一 profile."""
    aliases = ["rtx_4090", "RTX4090", "rtx4090", "nvidia_RTX_4090", "RTX_4090"]
    base = get_hardware_profile("RTX_4090")
    for name in aliases:
        hw = get_hardware_profile(name)
        assert hw.peak_flops_bf16 == base.peak_flops_bf16
        assert hw.mem_bandwidth == base.mem_bandwidth


def test_rtx_4090_ridge_point_reasonable():
    """Ridge point (FLOP/byte) BF16 应在 ~160 量级 (165 TFLOPS / 1 TB/s)."""
    hw = get_hardware_profile("RTX_4090")
    # ridge_point = peak_flops / mem_bandwidth (efficiency 全 1.0 默认下)
    rp = hw.ridge_point
    # 165e12 / 1008e9 ≈ 163.7
    assert 150 < rp < 180


def test_rtx_4090_vs_h100_relative_specs():
    """合理性: 4090 BF16 比 H100 SXM 弱 ~6×, BW 比 H100 弱 ~3×."""
    rtx = get_hardware_profile("RTX_4090")
    h100 = get_hardware_profile("H100")
    # H100 SXM ~989 TFLOPS BF16 dense; 4090 ~165 → ratio ~6
    assert h100.peak_flops_bf16 / rtx.peak_flops_bf16 > 4
    # H100 SXM ~3.35 TB/s; 4090 ~1.008 TB/s → ratio ~3.3
    assert h100.mem_bandwidth / rtx.mem_bandwidth > 2.5
    # H100 NVLink ~900 GB/s, 4090 PCIe ~32 GB/s → ratio > 10×
    assert h100.intra_node_bandwidth / rtx.intra_node_bandwidth > 10
