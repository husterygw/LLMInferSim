"""estimate_distinct_experts + MoERoutingProfile 单测."""
from __future__ import annotations

import pytest

from llm_infer_sim.core.operators.routing import (
    MoERoutingProfile,
    estimate_distinct_experts,
)


def test_decode_single_token_returns_topk():
    """T=1, skew=0 → distinct == top_k (decode 边界)."""
    assert estimate_distinct_experts(tokens=1, top_k=8, num_experts=128) == 8.0


def test_large_tokens_approaches_num_experts():
    """T→∞, skew=0 → distinct → num_experts."""
    distinct = estimate_distinct_experts(tokens=4096, top_k=8, num_experts=128)
    assert distinct > 127.0   # 几乎 sweep 全部


def test_skew_1_returns_topk_regardless_of_tokens():
    """skew=1 → 永远只 hit top_k."""
    for t in (1, 128, 4096):
        assert estimate_distinct_experts(tokens=t, top_k=8, num_experts=128, skew=1.0) == 8.0


def test_zero_inputs_safe():
    assert estimate_distinct_experts(0, 8, 128) == 0.0
    assert estimate_distinct_experts(128, 0, 128) == 0.0
    assert estimate_distinct_experts(128, 8, 0) == 0.0


def test_skew_clipped_to_unit_range():
    """skew>1 / <0 都截到 [0, 1]."""
    a = estimate_distinct_experts(128, 8, 128, skew=1.0)
    b = estimate_distinct_experts(128, 8, 128, skew=2.0)
    assert a == b
    c = estimate_distinct_experts(128, 8, 128, skew=0.0)
    d = estimate_distinct_experts(128, 8, 128, skew=-0.5)
    assert c == d


def test_routing_profile_balanced():
    p = MoERoutingProfile.balanced()
    assert p.distribution == "balanced"
    assert p.power_law_alpha == 0.0
    assert p.skew == 0.0


def test_routing_profile_power_law():
    p = MoERoutingProfile.power_law(alpha=1.2, skew=0.3)
    assert p.distribution == "power_law"
    assert p.power_law_alpha == 1.2
    assert p.skew == 0.3


def test_layer_skews_override_global():
    p = MoERoutingProfile(
        distribution="balanced", power_law_alpha=0.0,
        skew=0.0, layer_skews=(0.1, 0.5, 0.0),
    )
    assert p.get_skew_for_layer(0) == 0.1
    assert p.get_skew_for_layer(1) == 0.5
    assert p.get_skew_for_layer(2) == 0.0
    # 越界回 global
    assert p.get_skew_for_layer(3) == 0.0


def test_routing_profile_is_frozen():
    import dataclasses
    p = MoERoutingProfile.balanced()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.skew = 0.5  # type: ignore[misc]
