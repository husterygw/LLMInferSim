"""Collective case generation — AllReduce / AllToAll, profile-derived size."""
from __future__ import annotations

from collector.cases import collective
from collector.profiles._dims import ProfileSpec
from collector.schemas import OpKind


def _dense(**overrides) -> ProfileSpec:
    base = dict(
        profile_name="dense", hidden=2048, num_heads=32, num_kv_heads=4,
        head_dim=128, intermediate=6144, num_layers=48, vocab=151936,
    )
    base.update(overrides)
    return ProfileSpec(**base)


def _moe(**overrides) -> ProfileSpec:
    base = dict(
        profile_name="moe", hidden=2048, num_heads=32, num_kv_heads=4,
        head_dim=128, intermediate=6144, num_layers=48, vocab=151936,
        has_moe=True, moe_num_experts=128, moe_top_k=8, moe_intermediate=768,
    )
    base.update(overrides)
    return ProfileSpec(**base)


# ---------------------------------------------------------------------------
# get_cases_for_profile
# ---------------------------------------------------------------------------

class TestGetCasesForProfile:
    def test_dense_only_allreduce(self):
        """dense profile 默认 only AllReduce, 没 AllToAll."""
        p = _dense()
        cases = collective.get_cases_for_profile(
            p, num_tokens_values=[128], num_gpus_values=[4],
            topology_hints=["concentrated"],
        
            execution_modes=["cudagraph"],
        )
        subtypes = {c.params["op_subtype"] for c in cases}
        assert subtypes == {"allreduce"}

    def test_moe_includes_alltoall(self):
        """MoE profile 默认 AllReduce + AllToAll 两种都生成."""
        p = _moe()
        cases = collective.get_cases_for_profile(
            p, num_tokens_values=[128], num_gpus_values=[4],
            topology_hints=["concentrated"],
        
            execution_modes=["cudagraph"],
        )
        subtypes = {c.params["op_subtype"] for c in cases}
        assert subtypes == {"allreduce", "alltoall"}

    def test_message_size_derivation(self):
        """size = tokens × hidden × bytes_per_elem."""
        p = _dense(hidden=2048)
        cases = collective.get_cases_for_profile(
            p, num_tokens_values=[128], num_gpus_values=[4],
            topology_hints=["concentrated"], dtypes=["bf16"],
        execution_modes=["cudagraph"],
        )
        ar = cases[0]
        assert ar.params["message_size_bytes"] == 128 * 2048 * 2

    def test_all_op_collective(self):
        p = _moe()
        cases = collective.get_cases_for_profile(p, num_tokens_values=[1],
                                                   num_gpus_values=[2],
            execution_modes=["cudagraph"])
        assert all(c.op_kind == OpKind.COLLECTIVE for c in cases)

    def test_all_multi_gpu(self):
        """collective case 必须 multi_gpu=True (主 scheduler skip)."""
        p = _moe()
        cases = collective.get_cases_for_profile(p, num_tokens_values=[1],
                                                   num_gpus_values=[2],
            execution_modes=["cudagraph"])
        assert all(c.multi_gpu for c in cases)

    def test_topology_in_params(self):
        p = _dense()
        cases = collective.get_cases_for_profile(
            p, num_tokens_values=[1], num_gpus_values=[2],
            topology_hints=["concentrated", "balanced"],
        
            execution_modes=["cudagraph"],
        )
        tops = {c.params["topology_hint"] for c in cases}
        assert tops == {"concentrated", "balanced"}

    def test_num_gpus_in_params(self):
        p = _dense()
        cases = collective.get_cases_for_profile(
            p, num_tokens_values=[1], num_gpus_values=[2, 4, 8],
            topology_hints=["concentrated"],
        
            execution_modes=["cudagraph"],
        )
        gpus = {c.params["num_gpus"] for c in cases}
        assert gpus == {2, 4, 8}

    def test_alltoall_skipped_when_ep_exceeds_experts(self):
        """num_gpus > moe_num_experts 时 alltoall 跳过."""
        p = _moe(moe_num_experts=4)
        cases = collective.get_cases_for_profile(
            p, num_tokens_values=[1], num_gpus_values=[2, 4, 8],
            topology_hints=["concentrated"],
        
            execution_modes=["cudagraph"],
        )
        alltoalls = [c for c in cases if c.params["op_subtype"] == "alltoall"]
        gpus = {c.params["num_gpus"] for c in alltoalls}
        assert gpus == {2, 4}    # 8 > 4 experts → skip

    def test_no_model_no_profile_in_params(self):
        p = _moe(profile_name="profile_x")
        cases = collective.get_cases_for_profile(
            p, num_tokens_values=[1], num_gpus_values=[2],
            topology_hints=["concentrated"],
        
            execution_modes=["cudagraph"],
        )
        for c in cases:
            assert "model" not in c.params
            assert "profile_name" not in c.params
            assert "profile_x" not in c.case_id

    def test_default_sweep_count(self):
        """MoE 默认: 2 op × 3 ngpu × 1 dtype × 9 tokens × 2 topology = 108."""
        p = _moe()
        cases = collective.get_cases_for_profile(p,
            execution_modes=["cudagraph"])
        # AR: 3 × 1 × 9 × 2 = 54; AT: same 54; total 108
        assert len(cases) == 108


# ---------------------------------------------------------------------------
# Multi-profile dedup
# ---------------------------------------------------------------------------

class TestMultiProfileDedup:
    def test_same_hidden_same_size_dedup(self):
        """两个 profile hidden 一样 → AllReduce size 一样 → case_id 同 → dedup."""
        p1 = _dense(profile_name="p1", hidden=2048)
        p2 = _dense(profile_name="p2", hidden=2048)
        cases, sources = collective.get_cases(
            [p1, p2],
            num_tokens_values=[128], num_gpus_values=[4],
            topology_hints=["concentrated"],
        
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 1     # AR only, 1 size → 1 case
        for c in cases:
            assert sources[c.case_id] == ["p1", "p2"]

    def test_different_hidden_no_dedup(self):
        p1 = _dense(profile_name="p1", hidden=2048)
        p2 = _dense(profile_name="p2", hidden=2560)
        cases, _ = collective.get_cases(
            [p1, p2],
            num_tokens_values=[128], num_gpus_values=[4],
            topology_hints=["concentrated"],
        
            execution_modes=["cudagraph"],
        )
        assert len(cases) == 2     # 不同 hidden → 不同 size → 不 dedup

    def test_mixed_dense_moe_alltoall_only_from_moe(self):
        dense = _dense(profile_name="dense_p", hidden=2048)
        moe_p = _moe(profile_name="moe_p", hidden=2048)
        cases, sources = collective.get_cases(
            [dense, moe_p],
            num_tokens_values=[128], num_gpus_values=[4],
            topology_hints=["concentrated"],
        
            execution_modes=["cudagraph"],
        )
        ar_cases = [c for c in cases if c.params["op_subtype"] == "allreduce"]
        at_cases = [c for c in cases if c.params["op_subtype"] == "alltoall"]
        # AR: hidden=2048 同, 两 profile 共享 → 1 case, sources=[dense_p, moe_p]
        assert len(ar_cases) == 1
        assert sources[ar_cases[0].case_id] == ["dense_p", "moe_p"]
        # AT: 只 moe_p 贡献
        assert len(at_cases) == 1
        assert sources[at_cases[0].case_id] == ["moe_p"]


# ---------------------------------------------------------------------------
# 真 profile
# ---------------------------------------------------------------------------

class TestRealProfiles:
    def test_qwen3_4b_allreduce_only(self):
        from collector.profiles import qwen3_4b
        cases = collective.get_cases_for_profile(
            qwen3_4b.PROFILE,
            num_tokens_values=[128], num_gpus_values=[2, 4, 8],
            topology_hints=["concentrated", "balanced"],
        
            execution_modes=["cudagraph"],
        )
        # 3 ngpu × 2 topology × 1 token × 1 dtype = 6 (only AR)
        assert len(cases) == 6
        assert all(c.params["op_subtype"] == "allreduce" for c in cases)

    def test_qwen3_30b_a3b_includes_alltoall(self):
        from collector.profiles import qwen3_30b_a3b
        cases = collective.get_cases_for_profile(
            qwen3_30b_a3b.PROFILE,
            num_tokens_values=[128], num_gpus_values=[4],
            topology_hints=["concentrated"],
        
            execution_modes=["cudagraph"],
        )
        subtypes = {c.params["op_subtype"] for c in cases}
        assert subtypes == {"allreduce", "alltoall"}

    def test_two_qwen3_disjoint_hiddens(self):
        """Qwen3-4B hidden=2560 vs Qwen3-30B-A3B hidden=2048 → 不同 AR size → 不 dedup."""
        from collector.profiles import qwen3_4b, qwen3_30b_a3b
        cases, _ = collective.get_cases(
            [qwen3_4b.PROFILE, qwen3_30b_a3b.PROFILE],
            num_tokens_values=[128], num_gpus_values=[4],
            topology_hints=["concentrated"],
        
            execution_modes=["cudagraph"],
        )
        sizes = {c.params["message_size_bytes"] for c in cases
                 if c.params["op_subtype"] == "allreduce"}
        # 两个不同 size: 128×2560×2=655360, 128×2048×2=524288
        assert 128 * 2560 * 2 in sizes
        assert 128 * 2048 * 2 in sizes
