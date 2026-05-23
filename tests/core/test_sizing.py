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


def _large_moe_fixture() -> ModelConfig:
    """通用 MoE 配置, 用于 sizing dtype/EP-aware 测试 (改自原 V4-Flash-like)."""
    return ModelConfig(
        name="Large-MoE-fixture",
        hidden_dim=4096, num_heads=32, num_kv_heads=1, head_dim=128,
        ffn_dim=0, num_layers=43, vocab_size=129280,
        is_moe=True,
        num_experts=128, num_activated_experts=4, expert_dim=2048,
        num_shared_experts=1, moe_layer_freq=1, first_moe_layer=0,
    )


def test_explicit_expert_w_byte_halves_routed_bytes():
    """显式 expert_w_byte=0.5 时, routed expert 部分应按 0.5B/param 算, 非 expert 按 w_byte.

    (原 V4-Flash fp4 expert 用例; 现在通过显式参数表达, 不再依赖 model.expert_fp4.)
    """
    model = _large_moe_fixture()
    fp4_aware = estimate_param_bytes(model, w_byte=1.0, expert_w_byte=0.5)
    no_fp4 = estimate_param_bytes(model, w_byte=1.0, expert_w_byte=1.0)
    assert fp4_aware < no_fp4 * 0.6, \
        f"expert_w_byte=0.5 应明显省字节; fp4_aware={fp4_aware:,} no_fp4={no_fp4:,}"


def test_ep_shard_equals_tp_shard_when_ep_eq_tp():
    """ep=tp 时 per-rank 应跟纯 TP 等价 (vLLM FusedMoE EP/TP-only 存储等价)."""
    model = _large_moe_fixture()
    tp_only = per_rank_param_bytes(model, 1.0, tp_size=8, ep_size=1)
    ep_eq_tp = per_rank_param_bytes(model, 1.0, tp_size=8, ep_size=8)
    assert tp_only == ep_eq_tp


def test_base_w_byte_keeps_embed_lmhead_at_bf16_under_fp8():
    """FP8 模型 (w_byte=1.0) 但 lm_head/embed/final_norm 保留 bf16 (base=2.0).

    Qwen3-4B vocab=151936, hidden=2560: embed+lm_head = 2 × 151936 × 2560 = 778M params.
    base_w_byte=2.0 时 778M × 2 = 1.55 GB; 跟 w_byte=1.0 一起算只有 0.78 GB. 差 0.78GB.
    """
    model = _qwen3_4b()
    # 全 fp8 (旧行为, base 跟随 w)
    all_fp8 = estimate_param_bytes(model, w_byte=1.0, base_w_byte=1.0)
    # fp8 主体 + bf16 base (新行为)
    mixed = estimate_param_bytes(model, w_byte=1.0, base_w_byte=2.0)
    # 差 ≈ (embed + lm_head + norms) × 1 byte
    h = 2560
    embed_lmhead_norm_params = (
        2 * 151936 * h + h        # embed + lm_head + final_norm
        + 36 * 2 * h              # per-layer 2 RMSNorm
    )
    delta = mixed - all_fp8
    expected_delta = embed_lmhead_norm_params  # × (2-1) = × 1
    assert abs(delta - expected_delta) / expected_delta < 0.01, (
        f"base bf16 vs fp8 delta {delta} 偏离预期 {expected_delta}"
    )


def test_base_w_byte_none_falls_back_to_w_byte():
    """base_w_byte=None 时退化到 w_byte (向后兼容)."""
    model = _qwen3_4b()
    a = estimate_param_bytes(model, w_byte=1.0, base_w_byte=None)
    b = estimate_param_bytes(model, w_byte=1.0, base_w_byte=1.0)
    assert a == b


def test_per_rank_base_w_byte_under_fp8():
    """per_rank 也应用 base_w_byte 区分 lm_head/embed."""
    model = _qwen3_4b()
    fp8_only = per_rank_param_bytes(model, w_byte=1.0, tp_size=2)  # base=w default
    mixed = per_rank_param_bytes(
        model, w_byte=1.0, tp_size=2, base_w_byte=2.0,
    )
    assert mixed > fp8_only
    # 都 / tp=2 后差距也 / 2
    expected_delta_approx = (2 * 151936 * 2560 + 2560 + 36 * 2 * 2560) // 2
    actual_delta = mixed - fp8_only
    assert abs(actual_delta - expected_delta_approx) / expected_delta_approx < 0.02


def test_dp_does_not_change_dense_per_rank():
    """Pure DP (tp=1, dp=N): 每个 DP rank 是独立完整副本, weights/rank 不随 dp 变.

    sizing.per_rank_param_bytes 不接 dp_size 参数, 因为 dp 不影响 per-rank 切分;
    切分由 tp + ep 决定。`ep_size = tp × dp` (ParallelConfig.ep_size) 在 MoE+EP
    下隐式带入 dp 信息, dense / no-EP 时 ep=1 → dp 无影响, 这是正确语义。
    """
    model = _qwen3_4b()
    one_rank = per_rank_param_bytes(model, 2.0, tp_size=1, ep_size=1)
    full = estimate_param_bytes(model, 2.0)
    assert one_rank == full, "dp 不应切 dense weight"


def test_dp_plus_tp_dense_divides_by_tp_only():
    """DP+TP dense: tp=2 dp=4 → 每 rank weights = full / 2 (不 / 8)."""
    model = _qwen3_4b()
    full = estimate_param_bytes(model, 2.0)
    tp2 = per_rank_param_bytes(model, 2.0, tp_size=2, ep_size=1)
    # 允许 1% 误差 (int div 截断 + norm 处理)
    assert abs(tp2 - full // 2) / full < 0.01


def test_dp_plus_tp_plus_ep_moe_shards_routed_by_ep_world():
    """DP+TP+EP MoE: 例 tp=8 dp=2 ep=16 → routed expert / 16, dense / 8.

    Qwen3-235B 等多节点部署用此形态.
    """
    model = _large_moe_fixture()
    tp_only = per_rank_param_bytes(model, 1.0, tp_size=8, ep_size=1)
    # ep_size=16 模拟 tp=8 + dp=2 + enable_ep
    tp_plus_dp_ep = per_rank_param_bytes(model, 1.0, tp_size=8, ep_size=16)
    # MoE 主导, routed / 16 vs routed / 8: routed weight 部分应减半
    assert tp_plus_dp_ep < tp_only
    # 大致: 设 routed 占绝对主导, ratio ≈ 1/2 (16/8 倍切分)
    ratio = tp_plus_dp_ep / tp_only
    assert 0.4 < ratio < 0.95, f"ep_world 翻倍后 per-rank ratio {ratio} 应在 0.4-0.95"


def test_explicit_expert_w_byte_per_rank_matches_expected():
    """显式 expert_w_byte=0.5 路径 (原 V4-Flash fp4 case): fp4 vs fp8 routed expert 区分."""
    model = _large_moe_fixture()
    fp4_aware = per_rank_param_bytes(
        model, 1.0, tp_size=8, ep_size=8, expert_w_byte=0.5,
    ) / 1e9
    no_fp4 = per_rank_param_bytes(
        model, 1.0, tp_size=8, ep_size=8, expert_w_byte=1.0,
    ) / 1e9
    assert fp4_aware < no_fp4 * 0.6, f"expert_w_byte=0.5 应明显省字节; fp4={fp4_aware} no_fp4={no_fp4}"


def test_activation_scales_linearly():
    model = _qwen3_4b()
    a1 = estimate_activation_bytes(model, 1024, a_byte=2.0)
    a2 = estimate_activation_bytes(model, 2048, a_byte=2.0)
    assert a2 == pytest.approx(a1 * 2)
    a_fp32 = estimate_activation_bytes(model, 1024, a_byte=4.0)
    assert a_fp32 == pytest.approx(a1 * 2)
