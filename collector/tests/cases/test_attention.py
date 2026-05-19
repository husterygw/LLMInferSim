"""Attention case generation — 单 profile + 多 profile dedup."""
from __future__ import annotations

from collector.cases import attention
from collector.profiles._dims import ProfileSpec
from collector.schemas import OpKind


def _dummy(**overrides) -> ProfileSpec:
    base = dict(
        profile_name="dummy", hidden=128, num_heads=4, num_kv_heads=2,
        head_dim=32, intermediate=256, num_layers=2, vocab=1000,
    )
    base.update(overrides)
    return ProfileSpec(**base)


class TestGetCasesForProfile:
    def test_prefill_only(self):
        p = _dummy()
        cases = attention.get_cases_for_profile(
            p, prefill_isls=[128, 512], decode_batches=[1], decode_ctx_lens=[128],
            tp_sizes=[1], dtypes=["bf16"], include_prefill=True, include_decode=False,
        
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 2
        assert all(c.params["phase"] == "prefill" for c in cases)

    def test_decode_only(self):
        p = _dummy()
        cases = attention.get_cases_for_profile(
            p, prefill_isls=[128], decode_batches=[1, 4], decode_ctx_lens=[128, 512],
            tp_sizes=[1], dtypes=["bf16"], include_prefill=False, include_decode=True,
        
            execution_modes=["cudagraph"],
        )
        # 2 × 2 = 4
        assert len(cases) == 4

    def test_head_info_in_params(self):
        p = _dummy(num_heads=8, num_kv_heads=2, head_dim=64)
        cases = attention.get_cases_for_profile(
            p, prefill_isls=[128], decode_batches=[1], decode_ctx_lens=[128],
            tp_sizes=[1], dtypes=["bf16"],
        execution_modes=["cudagraph"],
        )
        for c in cases:
            assert c.params["num_heads"] == 8
            assert c.params["num_kv_heads"] == 2
            assert c.params["head_dim"] == 64

    def test_case_id_no_profile_name(self):
        p = _dummy(profile_name="profile_y")
        cases = attention.get_cases_for_profile(
            p, prefill_isls=[128], decode_batches=[1], decode_ctx_lens=[128],
            tp_sizes=[1], dtypes=["bf16"],
        execution_modes=["cudagraph"],
        )
        for c in cases:
            assert "profile_y" not in c.case_id
            assert "model" not in c.params
            assert "profile_name" not in c.params

    def test_all_op_attention(self):
        p = _dummy()
        cases = attention.get_cases_for_profile(
            p, prefill_isls=[128], decode_batches=[1], decode_ctx_lens=[128],
            tp_sizes=[1], dtypes=["bf16"],
        execution_modes=["cudagraph"],
        )
        assert all(c.op_kind == OpKind.ATTENTION for c in cases)


class TestMultiProfileDedup:
    def test_same_head_dims_dedup(self):
        """两个 profile head 配置一样 → attention case dedup."""
        p1 = _dummy(profile_name="p1")
        p2 = _dummy(profile_name="p2")   # 同 head dims
        cases, sources = attention.get_cases(
            [p1, p2],
            prefill_isls=[128], decode_batches=[1], decode_ctx_lens=[128],
            tp_sizes=[1], dtypes=["bf16"],
        execution_modes=["cudagraph"],
        )
        # 1 prefill + 1 decode = 2
        assert len(cases) == 2
        for c in cases:
            assert sources[c.case_id] == ["p1", "p2"]

    def test_different_head_dims_no_dedup(self):
        p1 = _dummy(profile_name="p1", num_heads=8)
        p2 = _dummy(profile_name="p2", num_heads=16)
        cases, _ = attention.get_cases(
            [p1, p2],
            prefill_isls=[128], decode_batches=[1], decode_ctx_lens=[128],
            tp_sizes=[1], dtypes=["bf16"],
        execution_modes=["cudagraph"],
        )
        # 不同 num_heads → 不同 case_id → 4 cases
        assert len(cases) == 4


class TestRealProfiles:
    def test_qwen3_4b_and_30b_distinct(self):
        """Qwen3-4B (num_kv_heads=8) 跟 Qwen3-30B-A3B (=4) attention shape 不同, 不 dedup."""
        from collector.profiles import qwen3_4b, qwen3_30b_a3b
        cases, sources = attention.get_cases(
            [qwen3_4b.PROFILE, qwen3_30b_a3b.PROFILE],
            prefill_isls=[2048], decode_batches=[1], decode_ctx_lens=[2048],
            tp_sizes=[1], dtypes=["bf16"],
        execution_modes=["cudagraph"],
        )
        # 完全不交叠 → 2 + 2 = 4
        assert len(cases) == 4
        for c in cases:
            assert len(sources[c.case_id]) == 1
