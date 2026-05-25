"""MoE case generation — distribution + parallel + dedup.

moe_plan Phase 2: cases/moe.py 改 AIC 字段口径; 此测试同步更新.
"""
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
        assert moe.get_cases_for_profile(
            p, execution_modes=["cudagraph"],
        ) == []

    def test_minimal_moe_profile(self):
        p = _dummy_moe()
        cases = moe.get_cases_for_profile(
            p,
            num_tokens_values=[128],
            parallel_configs=[(4, 1)],
            distributions=["balanced"],
            moe_dtypes=["bfloat16"],
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 1
        c = cases[0]
        assert c.op_kind == OpKind.MOE
        # AIC-aligned 字段
        assert c.params["num_tokens"] == 128
        assert c.params["topk"] == 8
        assert c.params["num_experts"] == 128
        assert c.params["inter_size"] == 768
        assert c.params["hidden_size"] == 2048
        assert c.params["moe_tp_size"] == 4
        assert c.params["moe_ep_size"] == 1
        assert c.params["distribution"] == "balanced"
        assert c.params["moe_dtype"] == "bfloat16"

    def test_three_distributions(self):
        """balanced + power_law_1.01 + power_law_1.2 = 3 cases."""
        p = _dummy_moe()
        cases = moe.get_cases_for_profile(
            p,
            num_tokens_values=[128],
            parallel_configs=[(4, 1)],
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 3   # 3 distributions
        dists = {c.params["distribution"] for c in cases}
        assert dists == {"balanced", "power_law_1.01", "power_law_1.2"}

    def test_parallel_configs(self):
        """vLLM 限定: moe_tp>1 AND moe_ep>1 同时不支持, (4,4) 被自动跳过."""
        p = _dummy_moe()
        cases = moe.get_cases_for_profile(
            p,
            num_tokens_values=[1],
            parallel_configs=[(1, 1), (1, 4), (4, 1), (4, 4)],
            distributions=["balanced"],
            execution_modes=["cudagraph"],
        )
        # (4, 4) 被新规则跳, 剩 3 个
        assert len(cases) == 3
        configs = {(c.params["moe_tp_size"], c.params["moe_ep_size"]) for c in cases}
        assert configs == {(1, 1), (1, 4), (4, 1)}

    def test_ep_exceeds_experts_skipped(self):
        """moe_ep_size > num_experts 应跳过."""
        p = _dummy_moe(moe_num_experts=8)
        cases = moe.get_cases_for_profile(
            p,
            num_tokens_values=[1],
            parallel_configs=[(1, 8), (1, 16)],   # ep=8 OK, ep=16 跳
            distributions=["balanced"],
            execution_modes=["cudagraph"],
        )
        eps = {c.params["moe_ep_size"] for c in cases}
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
        cases = moe.get_cases_for_profile(
            p, execution_modes=["cudagraph"],
        )
        assert all(c.op_kind == OpKind.MOE for c in cases)

    def test_default_sweep_count(self):
        """默认 (moe_plan Phase 2):
        10 num_tokens × 2 parallel × 3 distribution × 1 dtype × 1 mode = 60."""
        p = _dummy_moe()
        cases = moe.get_cases_for_profile(p)
        assert len(cases) == 10 * 2 * 3 * 1 * 1   # = 60


# ---------------------------------------------------------------------------
# distribution 进 case_id (区分 balanced / power_law / alpha)
# ---------------------------------------------------------------------------

class TestDistributionInCaseId:
    def test_different_distribution_different_id(self):
        """balanced 跟 power_law_1.2 case_id 必须不同."""
        p = _dummy_moe()
        balanced = moe.get_cases_for_profile(
            p, num_tokens_values=[128], parallel_configs=[(4, 1)],
            distributions=["balanced"],
            execution_modes=["cudagraph"],
        )[0]
        pw = moe.get_cases_for_profile(
            p, num_tokens_values=[128], parallel_configs=[(4, 1)],
            distributions=["power_law_1.2"],
            execution_modes=["cudagraph"],
        )[0]
        assert balanced.case_id != pw.case_id

    def test_different_alpha_different_id(self):
        """power_law_1.01 vs power_law_1.2 case_id 不同."""
        p = _dummy_moe()
        a = moe.get_cases_for_profile(
            p, num_tokens_values=[128], parallel_configs=[(4, 1)],
            distributions=["power_law_1.01"],
            execution_modes=["cudagraph"],
        )[0]
        b = moe.get_cases_for_profile(
            p, num_tokens_values=[128], parallel_configs=[(4, 1)],
            distributions=["power_law_1.2"],
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
        assert len(cases) == 3   # 3 distributions, only from moe_a
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
            parallel_configs=[(4, 1), (1, 4)],
            distributions=["balanced", "power_law_1.2"],
            execution_modes=["cudagraph"],
        )
        # 2 parallel × 2 distribution × 1 num_tokens = 4
        assert len(cases) == 4
        for c in cases:
            assert c.params["hidden_size"] == 2048
            assert c.params["inter_size"] == 768
            assert c.params["topk"] == 8
            assert c.params["num_experts"] == 128

    def test_qwen3_4b_returns_empty(self):
        """dense profile 不产 MoE case."""
        from collector.profiles import qwen3_4b
        assert moe.get_cases_for_profile(
            qwen3_4b.PROFILE, execution_modes=["cudagraph"],
        ) == []

    def test_two_profiles_only_moe_contributes(self):
        from collector.profiles import qwen3_4b, qwen3_30b_a3b
        cases, sources = moe.get_cases(
            [qwen3_4b.PROFILE, qwen3_30b_a3b.PROFILE],
            num_tokens_values=[1],
            parallel_configs=[(4, 1)],
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 3   # 3 distributions, only from qwen3_30b_a3b
        for c in cases:
            assert sources[c.case_id] == ["qwen3_30b_a3b"]
