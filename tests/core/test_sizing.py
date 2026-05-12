"""阶段 4 (4-ε): sizing helper 参数量 / 激活估算 (详设 §4.3.3)。

覆盖:
  - dense GQA (Qwen3-4B): total params ≈ 4B
  - dense MHA (opt-125m): total params ≈ 100M+
  - MoE (Qwen3-30B-A3B): routed experts 主导
  - TP shard: per_rank = total / tp
  - 激活估算公式正确
"""
import pytest

from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.profiles.sizing import (
    estimate_activation_bytes,
    estimate_param_bytes,
    estimate_param_count,
    per_rank_param_bytes,
)


def _qwen3_4b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-4B",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def _opt125m() -> ModelConfig:
    return ModelConfig(
        name="opt-125m",
        hidden_dim=768, num_heads=12, num_kv_heads=12, head_dim=64,
        ffn_dim=3072, num_layers=12, vocab_size=50272,
    )


def _qwen3_30b_a3b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-30B-A3B",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=0, num_layers=48, vocab_size=151936,
        is_moe=True,
        num_experts=128, num_activated_experts=8, expert_dim=768,
        num_shared_experts=0, moe_layer_freq=1, first_moe_layer=0,
    )


def test_qwen3_4b_param_count_near_4b():
    """Qwen3-4B 真实 ≈ 4.02B 参数, 估算误差应在 ±15% 内。"""
    n = estimate_param_count(_qwen3_4b())
    assert 3.4e9 < n < 4.6e9, f"got {n:.3e}"


def test_opt_125m_param_count_in_expected_range():
    """opt-125m 真实 125M, 估算因含独立 embed/lm_head 会偏大。"""
    n = estimate_param_count(_opt125m())
    assert 1.0e8 < n < 2.5e8, f"got {n:.3e}"


def test_moe_dominated_by_routed_experts():
    """Qwen3-30B-A3B 总参 ≈ 30B, 由 routed experts 主导。"""
    n = estimate_param_count(_qwen3_30b_a3b())
    assert 2.5e10 < n < 3.5e10, f"got {n:.3e}"


def test_param_bytes_scales_with_w_byte():
    model = _qwen3_4b()
    fp16 = estimate_param_bytes(model, w_byte=2.0)
    int8 = estimate_param_bytes(model, w_byte=1.0)
    int4 = estimate_param_bytes(model, w_byte=0.5)
    assert fp16 == 2 * int8 == 4 * int4


def test_per_rank_scales_with_tp():
    model = _qwen3_4b()
    total = estimate_param_bytes(model, 2.0)
    tp1 = per_rank_param_bytes(model, 2.0, 1)
    tp2 = per_rank_param_bytes(model, 2.0, 2)
    tp4 = per_rank_param_bytes(model, 2.0, 4)
    assert tp1 == total
    assert tp2 == total // 2
    assert tp4 == total // 4


def test_activation_scales_linearly():
    model = _qwen3_4b()
    a1 = estimate_activation_bytes(model, 1024, a_byte=2.0)
    a2 = estimate_activation_bytes(model, 2048, a_byte=2.0)
    assert a2 == pytest.approx(a1 * 2)
    a_fp32 = estimate_activation_bytes(model, 1024, a_byte=4.0)
    assert a_fp32 == pytest.approx(a1 * 2)
