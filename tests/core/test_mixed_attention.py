"""阶段 3.5: MixedAttentionEstimator 数值与一致性测试 (详设 §4.7.1b)。

覆盖:
  1. split_kernels 与 unified_ragged 都返回合法 dict 与正时间
  2. unified_ragged merged op flops/bytes = 子段 flops/bytes 之和 (一致性)
  3. unified vs split: roofline 取 max 次序不同, 时间数字应不同 (典型 mixed batch)
  4. 空 workload → 0 时间
  5. ragged_efficiency 反比例影响 unified_ragged 时间
  6. chunked_prefill_interleaved / decode_priority_prefill_append 仍 raise
"""
from types import SimpleNamespace

import pytest

from llm_infer_sim.core.cost_model.mixed_attention import (
    MixedAttentionEstimator,
    _merge_ops,
)
from llm_infer_sim.core.profiles.backend_profile import (
    BackendExecutionProfile,
    MixedAttentionPolicy,
)
from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle


def _make_vllm_config(hf, model_id="dummy"):
    """与 test_cost_model 一致: 不含 attention_config (自动 None → flash_attn_auto)。"""
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model=model_id),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1,
        ),
    )


@pytest.fixture
def qwen3_4b_bundle():
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=2560, num_hidden_layers=36,
        intermediate_size=9728, vocab_size=151936, head_dim=128,
    )
    return extract_profile_bundle(_make_vllm_config(hf, "Qwen3-4B"))


def _make_estimator(bundle, mode: str, ragged_efficiency: float = 1.0):
    backend = BackendExecutionProfile(
        name=f"test_{mode}",
        mixed_attention=MixedAttentionPolicy(
            mode=mode,
            ragged_efficiency=ragged_efficiency,
        ),
    )
    return MixedAttentionEstimator(
        model=bundle.model,
        hw=bundle.hw,
        deploy=bundle.deploy,
        backend=backend,
    )


# 典型 mixed step: 1 个 prefill 请求 (200 tokens) + 4 个 decode (ctx=512)
_TYPICAL_MIXED = dict(
    num_prefill_tokens=200,
    num_prefill_requests=1,
    num_decode_requests=4,
    max_prefill_seqlen=200,
    avg_decode_context_len=512,
)


def test_split_kernels_returns_positive_time(qwen3_4b_bundle):
    est = _make_estimator(qwen3_4b_bundle, "split_kernels")
    result = est.estimate(**_TYPICAL_MIXED)
    assert result["strategy"] == "split_kernels"
    assert result["per_layer_time"] > 0
    assert result["total_time"] == pytest.approx(
        result["per_layer_time"] * qwen3_4b_bundle.model.num_layers
    )
    assert {"t_prefill", "t_decode", "sync_overhead", "overlap"} <= set(
        result["breakdown"].keys()
    )


def test_unified_ragged_returns_positive_time(qwen3_4b_bundle):
    est = _make_estimator(qwen3_4b_bundle, "unified_ragged")
    result = est.estimate(**_TYPICAL_MIXED)
    assert result["strategy"] == "unified_ragged"
    assert result["per_layer_time"] > 0
    bd = result["breakdown"]
    assert bd["merged_flops"] > 0
    assert bd["merged_mem_bytes"] > 0
    assert bd["t_compute"] >= 0 and bd["t_memory"] >= 0
    assert bd["ragged_efficiency"] == pytest.approx(1.0)


def test_unified_ragged_merged_op_consistency(qwen3_4b_bundle):
    """merged op 的 flops / 5-way mem decomposition 必须 = 子段之和。"""
    est = _make_estimator(qwen3_4b_bundle, "unified_ragged")
    prefill_ops = est._build_prefill_ops(200, 1, 200)
    decode_ops = est._build_decode_ops(4, 512)
    all_ops = list(prefill_ops) + list(decode_ops)
    merged = _merge_ops(all_ops, name="test_merge")

    assert merged.flops == sum(op.flops for op in all_ops)
    assert merged.load_weight == sum(op.load_weight for op in all_ops)
    assert merged.load_act == sum(op.load_act for op in all_ops)
    assert merged.store_act == sum(op.store_act for op in all_ops)
    assert merged.load_kv_cache == sum(op.load_kv_cache for op in all_ops)
    assert merged.store_kv_cache == sum(op.store_kv_cache for op in all_ops)
    assert merged.mem_bytes == sum(op.mem_bytes for op in all_ops)
    assert merged.op_category == "attention"


def test_unified_equals_split_when_both_same_bound(qwen3_4b_bundle):
    """阶段 0-9 placeholder 哲学 (sync=0, efficiency=1) 下: 两段同 bound 时,
    split = max(c_pf, m_pf) + max(c_dc, m_dc) 与 unified = max(Σc, Σm) 数学恒等。

    Qwen3-4B + 200 prefill + 4 decode(ctx=512) 实测两段都 memory-bound, 应严格相等。
    """
    est_split = _make_estimator(qwen3_4b_bundle, "split_kernels")
    est_unified = _make_estimator(qwen3_4b_bundle, "unified_ragged")
    t_split = est_split.estimate(**_TYPICAL_MIXED)["per_layer_time"]
    t_unified = est_unified.estimate(**_TYPICAL_MIXED)["per_layer_time"]
    assert t_split > 0 and t_unified > 0
    assert t_split == pytest.approx(t_unified, rel=1e-9)


def test_unified_le_split_always(qwen3_4b_bundle):
    """阶段 0-9 placeholder 哲学下数学恒等式: unified <= split 永远成立。

    证明: max(a+c, b+d) <= max(a,b) + max(c,d) 对任意正数恒成立 (经典不等式)。
    取 a=t_c_pf, b=t_m_pf, c=t_c_dc, d=t_m_dc 即可。
    """
    workloads = [
        _TYPICAL_MIXED,
        # 长 prefill + 短 decode: prefill compute-bound 可能性高
        dict(num_prefill_tokens=8192, num_prefill_requests=1,
             num_decode_requests=4, max_prefill_seqlen=8192,
             avg_decode_context_len=512),
        # 短 prefill + 大量 long-ctx decode: decode memory-bound 显著
        dict(num_prefill_tokens=128, num_prefill_requests=1,
             num_decode_requests=32, max_prefill_seqlen=128,
             avg_decode_context_len=8192),
    ]
    est_split = _make_estimator(qwen3_4b_bundle, "split_kernels")
    est_unified = _make_estimator(qwen3_4b_bundle, "unified_ragged")
    for wl in workloads:
        t_split = est_split.estimate(**wl)["per_layer_time"]
        t_unified = est_unified.estimate(**wl)["per_layer_time"]
        assert t_unified <= t_split + 1e-15, f"violation @ {wl}: split={t_split}, unified={t_unified}"


def test_unified_ragged_empty_workload(qwen3_4b_bundle):
    est = _make_estimator(qwen3_4b_bundle, "unified_ragged")
    result = est.estimate(
        num_prefill_tokens=0, num_prefill_requests=0,
        num_decode_requests=0, max_prefill_seqlen=0,
        avg_decode_context_len=0,
    )
    assert result["per_layer_time"] == 0.0
    assert result["total_time"] == 0.0
    assert result["breakdown"].get("empty") is True


def test_ragged_efficiency_scales_inversely(qwen3_4b_bundle):
    """eff=0.5 应给出约 2× eff=1.0 的 per_layer_time。"""
    t1 = _make_estimator(qwen3_4b_bundle, "unified_ragged", ragged_efficiency=1.0)\
        .estimate(**_TYPICAL_MIXED)["per_layer_time"]
    t_half = _make_estimator(qwen3_4b_bundle, "unified_ragged", ragged_efficiency=0.5)\
        .estimate(**_TYPICAL_MIXED)["per_layer_time"]
    assert t_half == pytest.approx(t1 * 2.0, rel=1e-6)


def test_unimplemented_modes_still_raise(qwen3_4b_bundle):
    for mode in ("chunked_prefill_interleaved", "decode_priority_prefill_append"):
        est = _make_estimator(qwen3_4b_bundle, mode)
        with pytest.raises(NotImplementedError, match="§10.5"):
            est.estimate(**_TYPICAL_MIXED)


def test_unknown_mode_raises_value_error(qwen3_4b_bundle):
    est = _make_estimator(qwen3_4b_bundle, "nonexistent_mode")
    with pytest.raises(ValueError, match="Unknown mixed_attention.mode"):
        est.estimate(**_TYPICAL_MIXED)


def test_unified_decode_only_path(qwen3_4b_bundle):
    """纯 decode workload (无 prefill) 应有正常时间且不报错。"""
    est = _make_estimator(qwen3_4b_bundle, "unified_ragged")
    result = est.estimate(
        num_prefill_tokens=0, num_prefill_requests=0,
        num_decode_requests=8, max_prefill_seqlen=0,
        avg_decode_context_len=256,
    )
    assert result["per_layer_time"] > 0
    assert result["breakdown"]["merged_flops"] > 0


def test_unified_prefill_only_path(qwen3_4b_bundle):
    """纯 prefill workload (无 decode) 应有正常时间。"""
    est = _make_estimator(qwen3_4b_bundle, "unified_ragged")
    result = est.estimate(
        num_prefill_tokens=512, num_prefill_requests=1,
        num_decode_requests=0, max_prefill_seqlen=512,
        avg_decode_context_len=0,
    )
    assert result["per_layer_time"] > 0
    assert result["breakdown"]["merged_flops"] > 0
