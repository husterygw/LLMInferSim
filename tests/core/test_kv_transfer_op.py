"""KV transfer cost helper — 详设 §7.6."""
from __future__ import annotations

import pytest

from llm_infer_sim.core.ops.kv_transfer import kv_transfer_time


def test_zero_bytes_returns_just_latency():
    """空 send 也有 handshake latency。"""
    assert kv_transfer_time(0, bandwidth_gbps=25.0, latency_us=5.0) == pytest.approx(5e-6)


def test_pure_bandwidth_term():
    """1 GB @ 25 GB/s = 40 ms + 5 us startup ≈ 40.005 ms."""
    t = kv_transfer_time(int(1e9), bandwidth_gbps=25.0, latency_us=5.0)
    assert t == pytest.approx(1e9 / 25e9 + 5e-6, rel=1e-3)


def test_zero_bandwidth_inf():
    assert kv_transfer_time(1024, bandwidth_gbps=0.0, latency_us=1.0) == float("inf")


def test_realistic_v3_prefill_send():
    """DeepSeek-V3 prefill 1000 tok 的 KV transfer 估算.

    V3 MLA: 61 层 × 16 tok/block × (512+64) head_size × 2B = 1.12 MB/block
    1000 tok = 63 block → ~70.7 MB
    @ 25 GB/s = 2.83 ms + 5 us startup
    """
    block_bytes = 16 * 61 * (512 + 64) * 2  # 1,124,352
    blocks = 63
    bytes_total = blocks * block_bytes  # ~70.8 MB
    t = kv_transfer_time(bytes_total, bandwidth_gbps=25.0, latency_us=5.0)
    expected = bytes_total / 25e9 + 5e-6
    assert t == pytest.approx(expected, rel=1e-3)
    # 合理性: ms 量级, < 10ms 单请求
    assert t * 1e3 < 10.0
