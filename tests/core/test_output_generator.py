"""FakeTokenGenerator — fixed / deterministic_hash 两种模式 (详设 §4.3.5)。

覆盖:
  1. fixed 模式恒 emit fixed_token_id
  2. deterministic_hash 同 (prompt, num_generated) → 同 token (跨进程稳定)
  3. deterministic_hash 不同 prompt → 大概率不同 token (碰撞率合理)
  4. deterministic_hash 同 prompt 不同 num_generated → 不同 token (避免恒同输出)
  5. fixed_token_id 与 vocab_size clamp 行为
  6. from_env 走 LLM_INFER_SIM_FAKE_TOKEN_MODE
  7. 非法 mode raise
"""
from __future__ import annotations

import os
from unittest import mock

import pytest

from llm_infer_sim.core.simulation.output_generator import FakeTokenGenerator


def test_fixed_mode_emits_constant():
    gen = FakeTokenGenerator(mode="fixed", vocab_size=32000, fixed_token_id=7)
    for n in range(20):
        assert gen.next_token([1, 2, 3], num_generated=n) == 7


def test_fixed_mode_ignores_prompt():
    gen = FakeTokenGenerator(mode="fixed", vocab_size=32000, fixed_token_id=3)
    assert gen.next_token([10, 20, 30], 0) == 3
    assert gen.next_token([99, 88], 0) == 3


def test_fixed_token_id_clamped_to_vocab():
    """fixed_token_id 超 vocab_size 应被 clamp, 不能直接 emit 让 vLLM 报 detokenize 错。"""
    gen = FakeTokenGenerator(mode="fixed", vocab_size=100, fixed_token_id=999)
    assert 0 <= gen.next_token([1], 0) < 100


def test_deterministic_hash_same_input_same_output():
    gen = FakeTokenGenerator(mode="deterministic_hash", vocab_size=32000)
    a = gen.next_token([10, 11, 12], num_generated=0)
    b = gen.next_token([10, 11, 12], num_generated=0)
    assert a == b


def test_deterministic_hash_stable_across_instances():
    """跨 instance / 跨进程 (假设没 monkey-patch md5) 都同。"""
    g1 = FakeTokenGenerator(mode="deterministic_hash", vocab_size=50257)
    g2 = FakeTokenGenerator(mode="deterministic_hash", vocab_size=50257)
    assert g1.next_token([7, 8, 9], 3) == g2.next_token([7, 8, 9], 3)


def test_deterministic_hash_changes_with_num_generated():
    """同 prompt 不同步数: 避免输出恒同 token, 防止 prefix caching 假命中。"""
    gen = FakeTokenGenerator(mode="deterministic_hash", vocab_size=32000)
    seen = set()
    for n in range(10):
        seen.add(gen.next_token([1, 2, 3], num_generated=n))
    # 10 步生成至少 5 个不同 token (随机分布期望接近 10, 严格要求 ≥5)
    assert len(seen) >= 5, f"deterministic_hash 不应输出几乎恒定 token, 实测 {seen}"


def test_deterministic_hash_different_prompts_low_collision():
    """不同 prompt 在 200 次采样里碰撞率 < 5% (生日界限近似检查)。"""
    gen = FakeTokenGenerator(mode="deterministic_hash", vocab_size=32000)
    tokens = [gen.next_token([i, i + 1, i + 2], 0) for i in range(200)]
    unique = len(set(tokens))
    assert unique >= 190, f"碰撞过多: 200 prompts → 仅 {unique} unique"


def test_deterministic_hash_respects_vocab_bound():
    gen = FakeTokenGenerator(mode="deterministic_hash", vocab_size=128)
    for n in range(100):
        tok = gen.next_token([42, n], n)
        assert 0 <= tok < 128


def test_from_env_default_fixed():
    """LLM_INFER_SIM_FAKE_TOKEN_MODE 未设 → fixed (向后兼容)。"""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LLM_INFER_SIM_FAKE_TOKEN_MODE", None)
        gen = FakeTokenGenerator.from_env(vocab_size=1000)
        assert gen.mode == "fixed"


def test_from_env_deterministic_hash():
    with mock.patch.dict(
        os.environ, {"LLM_INFER_SIM_FAKE_TOKEN_MODE": "deterministic_hash"}, clear=False
    ):
        gen = FakeTokenGenerator.from_env(vocab_size=1000)
        assert gen.mode == "deterministic_hash"


def test_from_env_case_insensitive():
    with mock.patch.dict(
        os.environ, {"LLM_INFER_SIM_FAKE_TOKEN_MODE": "  Deterministic_Hash  "},
        clear=False,
    ):
        gen = FakeTokenGenerator.from_env(vocab_size=1000)
        assert gen.mode == "deterministic_hash"


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode="):
        FakeTokenGenerator(mode="bogus", vocab_size=1000)


def test_invalid_vocab_size_raises():
    with pytest.raises(ValueError, match="vocab_size"):
        FakeTokenGenerator(mode="fixed", vocab_size=0)
