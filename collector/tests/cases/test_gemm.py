"""GEMM case generation — 单 profile + 多 profile dedup."""
from __future__ import annotations

from collections import Counter

from collector.cases import gemm
from collector.profiles._dims import ProfileSpec
from collector.schemas import OpKind


def _dummy(**overrides) -> ProfileSpec:
    base = dict(
        profile_name="dummy", hidden=128, num_heads=4, num_kv_heads=2,
        head_dim=32, intermediate=256, num_layers=2, vocab=1000,
    )
    base.update(overrides)
    return ProfileSpec(**base)


# ---------------------------------------------------------------------------
# Single profile
# ---------------------------------------------------------------------------

class TestGetCasesForProfile:
    def test_dense_profile_includes_mlp(self):
        p = _dummy()
        cases = gemm.get_cases_for_profile(p, m_values=[1], tp_sizes=[1], dtypes=["bf16"],
            execution_modes=["cudagraph"])
        # 4 op (qkv/o/gate_up/down) + 1 lm_head = 5
        assert len(cases) == 5
        subtypes = {c.params["op_subtype"] for c in cases}
        assert subtypes == {"qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"}

    def test_moe_profile_excludes_mlp_by_default(self):
        p = _dummy(has_moe=True, moe_num_experts=8)
        cases = gemm.get_cases_for_profile(p, m_values=[1], tp_sizes=[1], dtypes=["bf16"],
            execution_modes=["cudagraph"])
        # MoE 模型自动跳 dense FFN → 2 op (qkv/o) + 1 lm_head = 3
        assert len(cases) == 3
        subtypes = {c.params["op_subtype"] for c in cases}
        assert "gate_up_proj" not in subtypes
        assert "down_proj" not in subtypes

    def test_include_mlp_explicit(self):
        p = _dummy(has_moe=True, moe_num_experts=8, intermediate=256)
        cases = gemm.get_cases_for_profile(
            p, m_values=[1], tp_sizes=[1], dtypes=["bf16"], include_mlp=True,
        
            execution_modes=["cudagraph"],
        )
        subtypes = {c.params["op_subtype"] for c in cases}
        assert "gate_up_proj" in subtypes

    def test_case_id_does_not_contain_profile_name(self):
        """case_id 应基于 op-level params, 不含 profile_name."""
        p = _dummy(profile_name="profile_x")
        cases = gemm.get_cases_for_profile(p, m_values=[1], tp_sizes=[1], dtypes=["bf16"],
            execution_modes=["cudagraph"])
        for c in cases:
            assert "profile_x" not in c.case_id
            assert "model" not in c.params

    def test_tp_shards_n(self):
        p = _dummy()
        c1 = gemm.get_cases_for_profile(p, m_values=[1], tp_sizes=[1],
                                        dtypes=["bf16"], include_lm_head=False,
            execution_modes=["cudagraph"])
        c2 = gemm.get_cases_for_profile(p, m_values=[1], tp_sizes=[2],
                                        dtypes=["bf16"], include_lm_head=False,
            execution_modes=["cudagraph"])
        qkv1 = [c for c in c1 if c.params["op_subtype"] == "qkv_proj"][0]
        qkv2 = [c for c in c2 if c.params["op_subtype"] == "qkv_proj"][0]
        assert qkv2.params["n"] == qkv1.params["n"] // 2

    def test_all_cases_op_gemm(self):
        p = _dummy()
        cases = gemm.get_cases_for_profile(p, m_values=[1], tp_sizes=[1], dtypes=["bf16"],
            execution_modes=["cudagraph"])
        assert all(c.op_kind == OpKind.GEMM for c in cases)


# ---------------------------------------------------------------------------
# Multi-profile dedup
# ---------------------------------------------------------------------------

class TestGetCasesMultiProfile:
    def test_two_profiles_same_shape_dedup(self):
        """两个 profile 出同 shape → case_id 一致 → 只保留一份."""
        p1 = _dummy(profile_name="p1")
        p2 = _dummy(profile_name="p2")     # same dims → same cases
        cases, sources = gemm.get_cases([p1, p2],
                                        m_values=[1], tp_sizes=[1], dtypes=["bf16"],
            execution_modes=["cudagraph"])
        # 5 cases from p1, 5 from p2, 全 dedup → 5
        assert len(cases) == 5
        # 每个 case 的 source_profiles 应包含两个
        for c in cases:
            assert sources[c.case_id] == ["p1", "p2"]

    def test_distinct_profiles_no_dedup(self):
        """不同 dims → 不同 case_id → 不 dedup."""
        p1 = _dummy(profile_name="p1", hidden=128)
        p2 = _dummy(profile_name="p2", hidden=256)
        cases, sources = gemm.get_cases([p1, p2],
                                        m_values=[1], tp_sizes=[1], dtypes=["bf16"],
            execution_modes=["cudagraph"])
        # 完全不交叠 → 5 + 5 = 10
        assert len(cases) == 10
        # 每个 case 的 source 只有一个 profile
        for c in cases:
            assert len(sources[c.case_id]) == 1

    def test_partial_overlap_attribution(self):
        """profile 部分 overlap 时, 同 shape 的 source 列两个 profile."""
        # p1 dense, p2 MoE 但 attention 维一致 → qkv/o/lm_head 一样, mlp 只 p1 有
        p1 = _dummy(profile_name="dense_p")
        p2 = _dummy(profile_name="moe_p", has_moe=True)   # 自动跳 mlp
        cases, sources = gemm.get_cases([p1, p2],
                                        m_values=[1], tp_sizes=[1], dtypes=["bf16"],
            execution_modes=["cudagraph"])
        # 共有: qkv, o, lm_head (共 3); 仅 p1: gate_up, down (共 2)
        # 合计 5
        assert len(cases) == 5
        # qkv 应被两个 profile 共享
        qkv = [c for c in cases if c.params["op_subtype"] == "qkv_proj"][0]
        assert sources[qkv.case_id] == ["dense_p", "moe_p"]
        # gate_up 只 dense_p 有
        gu = [c for c in cases if c.params["op_subtype"] == "gate_up_proj"][0]
        assert sources[gu.case_id] == ["dense_p"]


# ---------------------------------------------------------------------------
# 实际两个 Qwen3 profile 一起跑
# ---------------------------------------------------------------------------

class TestRealProfiles:
    def test_qwen3_4b_alone(self):
        from collector.profiles import qwen3_4b
        cases = gemm.get_cases_for_profile(qwen3_4b.PROFILE,
                                            m_values=[1, 128], tp_sizes=[1], dtypes=["bf16"],
            execution_modes=["cudagraph"])
        # 2 m × 4 op + 2 lm_head = 10
        assert len(cases) == 10

    def test_qwen3_30b_a3b_alone_moe(self):
        """all-MoE 自动 include_mlp=False."""
        from collector.profiles import qwen3_30b_a3b
        cases = gemm.get_cases_for_profile(qwen3_30b_a3b.PROFILE,
                                            m_values=[1, 128], tp_sizes=[1], dtypes=["bf16"],
            execution_modes=["cudagraph"])
        # 2 m × 2 op (qkv/o) + 2 lm_head = 6
        assert len(cases) == 6

    def test_two_qwen3_combined(self):
        """两个真 profile 一起跑, lm_head shape 一致 (vocab+hidden ≠), qkv 不同 → 部分 dedup."""
        from collector.profiles import qwen3_4b, qwen3_30b_a3b
        cases, sources = gemm.get_cases(
            [qwen3_4b.PROFILE, qwen3_30b_a3b.PROFILE],
            m_values=[1], tp_sizes=[1], dtypes=["bf16"],
        execution_modes=["cudagraph"],
        )
        # qwen3_4b hidden=2560, qwen3_30b hidden=2048 → 大部分 shape 不一样, 不会 dedup
        ids = [c.case_id for c in cases]
        assert len(ids) == len(set(ids))   # 无重复


def test_no_model_in_case_params():
    """整体保障: 真 profile 派生的 cases 里 params 没 model 字段."""
    from collector.profiles import qwen3_4b
    cases = gemm.get_cases_for_profile(qwen3_4b.PROFILE,
                                        m_values=[1], tp_sizes=[1], dtypes=["bf16"],
            execution_modes=["cudagraph"])
    for c in cases:
        assert "model" not in c.params
        assert "profile_name" not in c.params
