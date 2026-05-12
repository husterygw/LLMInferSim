"""阶段 5-δ: estimate_distinct_experts + MoERoutingPolicy 边界 (详设 §4.7.1)。

覆盖:
  1. 边界值: tokens=1 / tokens=∞ / skew=0 / skew=1
  2. coupon collector 公式数值正确性
  3. 单调性: skew=0 时 distinct(T) 单调递增, 上界 num_experts
  4. MoERoutingPolicy.get_skew_for_layer 行为 (per-layer override vs 全局)
"""
import math

import pytest

from llm_infer_sim.core.cost_model.moe_routing import (
    MoERoutingPolicy,
    estimate_distinct_experts,
)


# ------- 边界正确性 -------

def test_tokens_zero_returns_zero():
    assert estimate_distinct_experts(0, 8, 128, 0.0) == 0.0


def test_negative_inputs_return_zero():
    assert estimate_distinct_experts(-1, 8, 128, 0.0) == 0.0
    assert estimate_distinct_experts(4, 0, 128, 0.0) == 0.0
    assert estimate_distinct_experts(4, 8, 0, 0.0) == 0.0


def test_single_token_uniform_returns_topk():
    """tokens=1 + uniform: 严格等于 top_k (decode 边界, 不能改)。"""
    distinct = estimate_distinct_experts(tokens=1, top_k=8, num_experts=128, skew=0.0)
    assert distinct == pytest.approx(8.0)


def test_large_tokens_uniform_converges_to_num_experts():
    """tokens → ∞: distinct → num_experts (全 sweep)。"""
    distinct = estimate_distinct_experts(tokens=10000, top_k=8, num_experts=128, skew=0.0)
    assert distinct == pytest.approx(128.0, rel=1e-6)


def test_skew_one_pins_to_topk():
    """skew=1: 任意 tokens 都返回 top_k (极端 imbalance, 永远 hot experts)。"""
    for T in (1, 10, 100, 10000):
        d = estimate_distinct_experts(tokens=T, top_k=8, num_experts=128, skew=1.0)
        assert d == pytest.approx(8.0), f"T={T}: got {d}"


def test_skew_zero_to_one_interpolates():
    """skew 在 0 / 1 之间应该线性插值。"""
    T, top_k, N = 100, 8, 128
    uniform = estimate_distinct_experts(T, top_k, N, skew=0.0)
    worst = estimate_distinct_experts(T, top_k, N, skew=1.0)
    half = estimate_distinct_experts(T, top_k, N, skew=0.5)
    assert half == pytest.approx((uniform + worst) / 2)


def test_skew_clipped_above_one_and_below_zero():
    """skew 超出 [0, 1] 应该被 clip。"""
    d_clip_high = estimate_distinct_experts(100, 8, 128, skew=1.5)
    d_one = estimate_distinct_experts(100, 8, 128, skew=1.0)
    assert d_clip_high == pytest.approx(d_one)
    d_clip_low = estimate_distinct_experts(100, 8, 128, skew=-0.5)
    d_zero = estimate_distinct_experts(100, 8, 128, skew=0.0)
    assert d_clip_low == pytest.approx(d_zero)


# ------- coupon collector 公式数值 -------

def test_uniform_formula_matches_handcheck_for_qwen3_30b_a3b_decode_batch_4():
    """手算: T=4, top_k=8, N=128
    p_miss = (1 - 8/128)^4 = 0.9375^4 ≈ 0.7725
    distinct = 128 × (1 - 0.7725) ≈ 29.118
    """
    distinct = estimate_distinct_experts(tokens=4, top_k=8, num_experts=128, skew=0.0)
    expected = 128 * (1 - 0.9375 ** 4)
    assert distinct == pytest.approx(expected, rel=1e-9)
    # 也验证 numeric range
    assert 28.0 < distinct < 30.0


def test_uniform_formula_matches_handcheck_for_qwen3_30b_a3b_prefill_chunk_128():
    """手算: T=128, top_k=8, N=128
    p_miss = (1 - 8/128)^128 = 0.9375^128 ≈ 0.000275
    distinct ≈ 128 × (1 - 0.000275) ≈ 127.96  ← 几乎全 sweep
    """
    distinct = estimate_distinct_experts(tokens=128, top_k=8, num_experts=128, skew=0.0)
    expected = 128 * (1 - 0.9375 ** 128)
    assert distinct == pytest.approx(expected, rel=1e-9)
    assert distinct > 127.0


# ------- 单调性 -------

def test_distinct_monotonic_in_tokens():
    """skew=0 时 distinct(T) 应该单调不降。"""
    top_k, N = 8, 128
    last = -1.0
    for T in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024):
        d = estimate_distinct_experts(T, top_k, N, skew=0.0)
        assert d >= last - 1e-9, f"non-monotonic at T={T}: {d} < {last}"
        last = d


def test_distinct_capped_at_num_experts():
    """无论多大 tokens, distinct 都不会超过 num_experts。"""
    for T in (1, 100, 1000, 10000, 100000):
        d = estimate_distinct_experts(T, 8, 128, skew=0.0)
        assert d <= 128.0 + 1e-9


# ------- MoERoutingPolicy -------

def test_policy_default_is_uniform():
    """默认 policy 必须 skew=0 (与 EfficiencyProfile.placeholder=1.0 哲学一致)。"""
    p = MoERoutingPolicy()
    assert p.skew == 0.0
    assert p.layer_skews is None
    assert p.use_trace is False
    # 任意层都返回全局 skew=0
    for layer_idx in (0, 5, 47):
        assert p.get_skew_for_layer(layer_idx) == 0.0


def test_policy_global_skew_returned_when_no_per_layer():
    p = MoERoutingPolicy(skew=0.3)
    assert p.get_skew_for_layer(0) == 0.3
    assert p.get_skew_for_layer(47) == 0.3


def test_policy_per_layer_skews_override_global():
    p = MoERoutingPolicy(skew=0.1, layer_skews=[0.0, 0.2, 0.5])
    assert p.get_skew_for_layer(0) == 0.0
    assert p.get_skew_for_layer(1) == 0.2
    assert p.get_skew_for_layer(2) == 0.5
    # 索引越界 fallback 到全局
    assert p.get_skew_for_layer(5) == 0.1
