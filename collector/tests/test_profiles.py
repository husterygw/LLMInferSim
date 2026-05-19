"""profiles/ — ProfileSpec + registry."""
from __future__ import annotations

import pytest

from collector.profiles import ProfileSpec
from collector.profiles import registry as profile_registry


class TestProfileSpec:
    def test_qwen3_4b_dims(self):
        from collector.profiles import qwen3_4b
        p = qwen3_4b.PROFILE
        assert p.profile_name == "qwen3_4b"
        assert p.hidden == 2560
        assert p.q_dim == 32 * 128         # 4096
        assert p.kv_dim == 8 * 128          # 1024
        assert p.qkv_out == 4096 + 2 * 1024  # 6144
        assert not p.has_moe

    def test_qwen3_30b_a3b_dims(self):
        from collector.profiles import qwen3_30b_a3b
        p = qwen3_30b_a3b.PROFILE
        assert p.profile_name == "qwen3_30b_a3b"
        assert p.hidden == 2048
        assert p.q_dim == 32 * 128         # 4096
        assert p.kv_dim == 4 * 128          # 512
        assert p.qkv_out == 4096 + 2 * 512   # 5120
        assert p.has_moe
        assert p.moe_num_experts == 128
        assert p.moe_top_k == 8


class TestProfileRegistry:
    def test_list_known(self):
        names = profile_registry.list_profile_names()
        assert "qwen3_4b" in names
        assert "qwen3_30b_a3b" in names

    def test_load_qwen3_4b(self):
        p = profile_registry.load_profile("qwen3_4b")
        assert isinstance(p, ProfileSpec)
        assert p.profile_name == "qwen3_4b"

    def test_load_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown profile"):
            profile_registry.load_profile("not_a_model")
