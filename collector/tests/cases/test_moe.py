"""MoE case generation — routing distribution + parallel + dedup."""
from __future__ import annotations

from collector.cases import moe
from collector.profiles._dims import ProfileSpec
from collector.schemas import OpKind


def _dummy_moe(**overrides) -> ProfileSpec:
    base = dict(
        profile_name="moe_dummy",
        hidden=2048,
        num_heads=32, num_kv_heads=4, head_dim=128,
        intermediate=6144, num_layers=48, vocab=151936,
        has_moe=True,
        moe_num_experts=128, moe_top_k=8, moe_intermediate=768,
    )
    base.update(overrides)
    return ProfileSpec(**base)


def _dummy_dense(**overrides) -> ProfileSpec:
    base = dict(
        profile_name="dense_dummy",
        hidden=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        intermediate=6144, num_layers=48, vocab=151936,
    )
    base.update(overrides)
    return ProfileSpec(**base)


# ---------------------------------------------------------------------------
# get_cases_for_profile
# ---------------------------------------------------------------------------

class TestGetCasesForProfile:
    def test_non_moe_profile_returns_empty(self):
        p = _dummy_dense()
        assert moe.get_cases_for_profile(p,
            execution_modes=["cudagraph"]) == []

    def test_minimal_moe_profile(self):
        p = _dummy_moe()
        cases = moe.get_cases_for_profile(
            p,
            num_tokens_values=[128],
            parallel_configs=[(4, 1)],
            routings=[("balanced", 0.0)],
            dtypes=["bf16"],
        execution_modes=["cudagraph"],
        )
        assert len(cases) == 1
        c = cases[0]
        assert c.op_kind == OpKind.MOE
        assert c.params["num_tokens"] == 128
        assert c.params["topk"] == 8
        assert c.params["num_experts"] == 128
        assert c.params["moe_intermediate"] == 768
        assert c.params["tp"] == 4
        assert c.params["ep"] == 1
        assert c.params["routing_distribution"] == "balanced"

    def test_three_routings(self):
        """balanced + power_law_1.01 + power_law_1.2 = 3 cases."""
        p = _dummy_moe()
        cases = moe.get_cases_for_profile(
            p,
            num_tokens_values=[128],
            parallel_configs=[(4, 1)],
        
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 3   # 3 routings
        routings = {(c.params["routing_distribution"], c.params["power_law_alpha"])
                    for c in cases}
        assert routings == {
            ("balanced", 0.0),
            ("power_law", 1.01),
            ("power_law", 1.2),
        }

    def test_parallel_configs(self):
        p = _dummy_moe()
        cases = moe.get_cases_for_profile(
            p,
            num_tokens_values=[1],
            parallel_configs=[(1, 1), (1, 4), (4, 1), (4, 4)],
            routings=[("balanced", 0.0)],
        
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 4
        configs = {(c.params["tp"], c.params["ep"]) for c in cases}
        assert configs == {(1, 1), (1, 4), (4, 1), (4, 4)}

    def test_ep_exceeds_experts_skipped(self):
        """EP > num_experts 应跳过."""
        p = _dummy_moe(moe_num_experts=8)
        cases = moe.get_cases_for_profile(
            p,
            num_tokens_values=[1],
            parallel_configs=[(1, 8), (1, 16)],   # EP=8 OK, EP=16 跳
            routings=[("balanced", 0.0)],
        
            execution_modes=["cudagraph"],
        )
        eps = {c.params["ep"] for c in cases}
        assert 8 in eps
        assert 16 not in eps

    def test_case_id_does_not_contain_profile_name(self):
        p = _dummy_moe(profile_name="profile_z")
        cases = moe.get_cases_for_profile(
            p,
            num_tokens_values=[1],
            parallel_configs=[(4, 1)],
        
            execution_modes=["cudagraph"],
        )
        for c in cases:
            assert "profile_z" not in c.case_id
            assert "model" not in c.params
            assert "profile_name" not in c.params

    def test_all_cases_op_moe(self):
        p = _dummy_moe()
        cases = moe.get_cases_for_profile(p,
            execution_modes=["cudagraph"])
        assert all(c.op_kind == OpKind.MOE for c in cases)

    def test_default_sweep_count(self):
        """默认: 9 num_tokens × 4 parallel × 3 routing × 1 dtype = 108."""
        p = _dummy_moe()
        cases = moe.get_cases_for_profile(p,
            execution_modes=["cudagraph"])
        assert len(cases) == 9 * 4 * 3 * 1   # 108


# ---------------------------------------------------------------------------
# routing distribution + alpha 进 case_id
# ---------------------------------------------------------------------------

class TestRoutingInCaseId:
    def test_different_routing_different_id(self):
        """balanced 跟 power_law alpha=1.2 case_id 必须不同."""
        p = _dummy_moe()
        balanced = moe.get_cases_for_profile(
            p, num_tokens_values=[128], parallel_configs=[(4, 1)],
            routings=[("balanced", 0.0)],
        
            execution_modes=["cudagraph"],
        )[0]
        pw = moe.get_cases_for_profile(
            p, num_tokens_values=[128], parallel_configs=[(4, 1)],
            routings=[("power_law", 1.2)],
        
            execution_modes=["cudagraph"],
        )[0]
        assert balanced.case_id != pw.case_id

    def test_different_alpha_different_id(self):
        """power_law alpha=1.01 vs 1.2 case_id 不同."""
        p = _dummy_moe()
        a = moe.get_cases_for_profile(
            p, num_tokens_values=[128], parallel_configs=[(4, 1)],
            routings=[("power_law", 1.01)],
        
            execution_modes=["cudagraph"],
        )[0]
        b = moe.get_cases_for_profile(
            p, num_tokens_values=[128], parallel_configs=[(4, 1)],
            routings=[("power_law", 1.2)],
        
            execution_modes=["cudagraph"],
        )[0]
        assert a.case_id != b.case_id


# ---------------------------------------------------------------------------
# Multi-profile dedup
# ---------------------------------------------------------------------------

class TestMultiProfileDedup:
    def test_dense_profile_contributes_nothing(self):
        """混入 dense profile 不产 MoE case, 不影响其他 MoE profile."""
        dense = _dummy_dense(profile_name="dense_a")
        moe_p = _dummy_moe(profile_name="moe_a")
        cases, sources = moe.get_cases(
            [dense, moe_p],
            num_tokens_values=[1],
            parallel_configs=[(4, 1)],
        
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 3   # 3 routings, only from moe_a
        for c in cases:
            assert sources[c.case_id] == ["moe_a"]

    def test_two_moe_profiles_same_dims_dedup(self):
        """两 MoE profile 同 dims → case_id 相同 → dedup, sources 列两个."""
        p1 = _dummy_moe(profile_name="p1")
        p2 = _dummy_moe(profile_name="p2")
        cases, sources = moe.get_cases(
            [p1, p2],
            num_tokens_values=[1],
            parallel_configs=[(4, 1)],
        
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 3   # 不是 6, dedup 了
        for c in cases:
            assert sources[c.case_id] == ["p1", "p2"]

    def test_different_moe_dims_no_dedup(self):
        """num_experts 不同 → case_id 不同 → 不 dedup."""
        p1 = _dummy_moe(profile_name="p1", moe_num_experts=64)
        p2 = _dummy_moe(profile_name="p2", moe_num_experts=128)
        cases, _ = moe.get_cases(
            [p1, p2],
            num_tokens_values=[1],
            parallel_configs=[(4, 1)],
        
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 6   # 3 + 3


# ---------------------------------------------------------------------------
# Qwen3-30B-A3B real profile
# ---------------------------------------------------------------------------

class TestQwen3_30B_A3B_Real:
    def test_get_cases_for_profile(self):
        from collector.profiles import qwen3_30b_a3b
        cases = moe.get_cases_for_profile(
            qwen3_30b_a3b.PROFILE,
            num_tokens_values=[128],
            parallel_configs=[(4, 1), (4, 4)],
            routings=[("balanced", 0.0), ("power_law", 1.2)],
        
            execution_modes=["cudagraph"],
        )
        # 2 parallel × 2 routing × 1 num_tokens = 4
        assert len(cases) == 4
        for c in cases:
            assert c.params["hidden"] == 2048
            assert c.params["moe_intermediate"] == 768
            assert c.params["topk"] == 8
            assert c.params["num_experts"] == 128

    def test_qwen3_4b_returns_empty(self):
        """dense profile 不产 MoE case."""
        from collector.profiles import qwen3_4b
        assert moe.get_cases_for_profile(qwen3_4b.PROFILE,
            execution_modes=["cudagraph"]) == []

    def test_two_profiles_only_moe_contributes(self):
        from collector.profiles import qwen3_4b, qwen3_30b_a3b
        cases, sources = moe.get_cases(
            [qwen3_4b.PROFILE, qwen3_30b_a3b.PROFILE],
            num_tokens_values=[1],
            parallel_configs=[(4, 1)],
        
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 3   # 3 routings, only from qwen3_30b_a3b
        for c in cases:
            assert sources[c.case_id] == ["qwen3_30b_a3b"]
