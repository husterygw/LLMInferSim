"""阶段 8-β: MLA cost 公式手算 vs actual 对照 (详设 §4.1.4)。

按记忆 feedback_cost_formula_handcheck.md 必做。

覆盖 5 个 MLA 核心 op 公式 (DeepSeek-V3 tp=8 decode tokens=1):
  1. q_a_proj:           h → q_lora_rank
  2. q_b_proj:           q_lora_rank → heads_per_tp × (qk_nope + qk_rope)
  3. kv_a_proj_with_mqa: h → kv_lora_rank + qk_rope  (output to KV cache)
  4. kv_b_proj:          kv_lora_rank → heads_per_tp × (qk_nope + v_head_dim)
  5. o_proj:             heads_per_tp × v_head_dim → h

防御性:
  - q_lora_rank == 0 时 fallback 到单 q_proj (没有 a/b 分解)
  - q_lora_rank > 0 时分解
  - v_head_dim 默认 = qk_nope_head_dim (V3 config 无显式 v_head_dim 字段)
  - q_proj/k_proj/v_proj/qkv_fused 在 MLA path 下 *都不应出现*
"""
from types import SimpleNamespace

import pytest

from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
from llm_infer_sim.core.cost_model.layer_builder import dense_layer_time


def _deepseek_v3_bundle(tp_size: int = 8):
    """DeepSeek-V3 真实 hf_config 字段构造 bundle (kv_lora=512, q_lora=1536)。"""
    hf = SimpleNamespace(
        model_type="deepseek_v3",
        num_attention_heads=128, num_key_value_heads=128,
        hidden_size=7168, num_hidden_layers=61,
        intermediate_size=18432, vocab_size=129280,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=1536,
        # MoE 字段 (本测试只看 attention block, MoE 由 test_moe_cost_consistency 覆盖)
        n_routed_experts=256, num_experts_per_tok=8,
        moe_intermediate_size=2048, n_shared_experts=1, first_k_dense_replace=3,
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="deepseek-ai/DeepSeek-V3"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, data_parallel_size=1,
            enable_expert_parallel=False,
        ),
    )
    return extract_profile_bundle(vc)


def _find_op(lr, name):
    for op in lr.ops:
        if op.name == name:
            return op
    raise AssertionError(f"op {name!r} not in {[o.name for o in lr.ops]}")


def _has_op(lr, name) -> bool:
    return any(op.name == name for op in lr.ops)


# ------- profile_extractor v_head_dim fallback -------

def test_v_head_dim_fallback_to_qk_nope():
    """V3 config 无 v_head_dim 字段, profile_extractor 应 fallback 到 qk_nope_head_dim=128
    (而不是旧代码的 head_dim=hidden/num_heads=56)。"""
    b = _deepseek_v3_bundle()
    assert b.model.v_head_dim == 128  # = qk_nope_head_dim
    # 防止退到 head_dim=56
    assert b.model.v_head_dim != b.model.hidden_dim // b.model.num_heads


def test_q_lora_rank_passthrough():
    """V3 q_lora_rank=1536 必须透传."""
    assert _deepseek_v3_bundle().model.q_lora_rank == 1536


# ------- MLA ops 手算 vs actual -------

def test_mla_layer_uses_q_a_q_b_not_single_q_proj():
    """V3 (q_lora_rank > 0) 必须用 q_a_proj + q_b_proj, 而不是单 q_proj 也不是 qkv_fused."""
    b = _deepseek_v3_bundle()
    lr = dense_layer_time(0, "decode", 1, 128, b.model, b.deploy, b.hw)
    assert _has_op(lr, "q_a_proj")
    assert _has_op(lr, "q_b_proj")
    assert not _has_op(lr, "q_proj")     # 没有单一 q_proj
    assert not _has_op(lr, "qkv_proj")   # 没有 fused QKV


def test_mla_layer_no_separate_k_v_proj():
    """MLA 路径不应有独立的 k_proj/v_proj, 而是 kv_a_proj_with_mqa (single fused)."""
    b = _deepseek_v3_bundle()
    lr = dense_layer_time(0, "decode", 1, 128, b.model, b.deploy, b.hw)
    assert _has_op(lr, "kv_a_proj_with_mqa")
    assert not _has_op(lr, "k_proj")
    assert not _has_op(lr, "v_proj")


def test_q_a_proj_handcheck():
    """q_a_proj: h → q_lora_rank, flops = 2 × tokens × h × q_lora_rank."""
    b = _deepseek_v3_bundle()
    m, deploy = b.model, b.deploy
    tokens = 1
    lr = dense_layer_time(0, "decode", tokens, 128, m, deploy, b.hw)
    op = _find_op(lr, "q_a_proj")
    expected = 2 * tokens * m.hidden_dim * m.q_lora_rank
    assert op.flops == expected  # 2*1*7168*1536 = 22,020,096


def test_q_b_proj_handcheck():
    """q_b_proj: q_lora_rank → heads_per_tp × (qk_nope + qk_rope)."""
    b = _deepseek_v3_bundle()
    m, deploy = b.model, b.deploy
    tokens = 1
    lr = dense_layer_time(0, "decode", tokens, 128, m, deploy, b.hw)
    op = _find_op(lr, "q_b_proj")
    heads_per_tp = m.num_heads // deploy.tp
    q_head_dim = m.qk_nope_head_dim + m.rope_head_dim
    expected = 2 * tokens * m.q_lora_rank * heads_per_tp * q_head_dim
    assert op.flops == expected  # 2*1*1536*16*192 = 9,437,184


def test_kv_a_proj_with_mqa_handcheck():
    """kv_a_proj_with_mqa: h → kv_lora_rank + qk_rope_head_dim, output 到 KV cache."""
    b = _deepseek_v3_bundle()
    m, deploy = b.model, b.deploy
    tokens = 1
    lr = dense_layer_time(0, "decode", tokens, 128, m, deploy, b.hw)
    op = _find_op(lr, "kv_a_proj_with_mqa")
    expected_oc = m.kv_lora_rank + m.rope_head_dim
    expected_flops = 2 * tokens * m.hidden_dim * expected_oc
    assert op.flops == expected_flops  # 2*1*7168*(512+64) = 8,257,536
    # 输出在 KV cache, 不在 activation
    assert op.store_kv_cache > 0
    assert op.store_act == 0


def test_kv_b_proj_handcheck():
    """kv_b_proj: kv_lora_rank → heads_per_tp × (qk_nope + v_head_dim), compute-time."""
    b = _deepseek_v3_bundle()
    m, deploy = b.model, b.deploy
    tokens = 1
    lr = dense_layer_time(0, "decode", tokens, 128, m, deploy, b.hw)
    op = _find_op(lr, "kv_b_proj")
    heads_per_tp = m.num_heads // deploy.tp
    expected_oc = heads_per_tp * (m.qk_nope_head_dim + m.v_head_dim)
    expected_flops = 2 * tokens * m.kv_lora_rank * expected_oc
    assert op.flops == expected_flops  # 2*1*512*16*(128+128) = 4,194,304
    # kv_b_proj 是 compute-time decompression, 输出在 activation 不在 KV cache
    assert op.store_act > 0
    assert op.store_kv_cache == 0


def test_o_proj_uses_v_head_dim_not_head_dim():
    """o_proj 的 input dim 应 = heads_per_tp × v_head_dim, 不是 heads_per_tp × head_dim.

    V3: v_head_dim=128, head_dim=hidden/num_heads=56. 用错维度差 ~2.3×。
    """
    b = _deepseek_v3_bundle()
    m, deploy = b.model, b.deploy
    tokens = 1
    lr = dense_layer_time(0, "decode", tokens, 128, m, deploy, b.hw)
    op = _find_op(lr, "o_proj")
    heads_per_tp = m.num_heads // deploy.tp
    expected_ic = heads_per_tp * m.v_head_dim
    expected_flops = 2 * tokens * expected_ic * m.hidden_dim
    assert op.flops == expected_flops  # 2*1*16*128*7168 = 29,360,128
    # 防御性: 如果用错 head_dim=56, flops 会差 ~2.3× (128/56)
    wrong_flops = 2 * tokens * heads_per_tp * m.head_dim * m.hidden_dim
    assert op.flops != wrong_flops


# ------- q_lora_rank == 0 路径(假定 V2-Lite 这类无 Q-LoRA 模型)-------

def test_mla_without_q_lora_uses_single_q_proj():
    """kv_lora_rank > 0 但 q_lora_rank == 0 时, Q 走单 q_proj (无 a/b 分解)."""
    hf = SimpleNamespace(
        model_type="deepseek_v3",
        num_attention_heads=16, num_key_value_heads=16,
        hidden_size=2048, num_hidden_layers=27,
        intermediate_size=10944, vocab_size=102400,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=0,       # ← 没有 Q-side LoRA
        n_routed_experts=64, num_experts_per_tok=6,
        moe_intermediate_size=1408, n_shared_experts=2, first_k_dense_replace=1,
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="fake-v2-lite"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, enable_expert_parallel=False,
        ),
    )
    bundle = extract_profile_bundle(vc)
    assert bundle.model.q_lora_rank == 0
    lr = dense_layer_time(0, "decode", 1, 128, bundle.model, bundle.deploy, bundle.hw)
    assert _has_op(lr, "q_proj")          # 单 q_proj
    assert not _has_op(lr, "q_a_proj")
    assert not _has_op(lr, "q_b_proj")
    # KV 仍然走 mqa fused
    assert _has_op(lr, "kv_a_proj_with_mqa")
    assert _has_op(lr, "kv_b_proj")


# ------- 修复前 vs 修复后总 flops 差异 -------

def test_total_attn_proj_flops_significantly_higher_than_pre_fix():
    """总 attention proj flops (5 个 op 之和) 应 ≥ 70M, 显著高于修复前的 ~42M.

    这条断言保证如果未来谁误改回 head_dim 维度 (旧 bug), test 立刻挂。
    """
    b = _deepseek_v3_bundle()
    lr = dense_layer_time(0, "decode", 1, 128, b.model, b.deploy, b.hw)
    proj_ops = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
    total = sum(_find_op(lr, n).flops for n in proj_ops)
    assert total > 70e6
    assert total < 80e6  # 也别异常高


# ------- MLA prefill flash KV IO (audit 修复后) -------

def test_mla_prefill_new_token_does_not_read_kv_cache():
    """MLA 新 token prefill 不应读 KV cache.

    vLLM 真实路径: prefill 时 K/V 来自上游 kv_b_proj staging activation, 不读 cache;
    只有 chunked prefill 跨 step 的 prior context 才读 c_kv (single-head latent).
    旧 bug: load_kv_cache = seqlen*(_qk+_v)*num_heads*kv_byte (虚高 num_heads≈128 倍).
    """
    b = _deepseek_v3_bundle(tp_size=8)
    # prefill 新 token: ctx_len == seq, prior_ctx = 0
    lr = dense_layer_time(0, "prefill", tokens=128, ctx_len=128,
                          model=b.model, deploy=b.deploy, hw=b.hw)
    fused = _find_op(lr, "fused_mla_attention")
    assert fused.load_kv_cache == 0, (
        f"MLA new-token prefill load_kv_cache should be 0, got {fused.load_kv_cache}")


def test_mla_prefill_kv_staging_in_load_act_not_kv_cache():
    """MLA prefill 的 K/V staging (kv_b_proj 输出) 必须进 load_act, 用 a_byte (非 kv_byte)。"""
    b = _deepseek_v3_bundle(tp_size=8)
    m, deploy, hw = b.model, b.deploy, b.hw
    lr = dense_layer_time(0, "prefill", tokens=128, ctx_len=128, model=m, deploy=deploy, hw=hw)
    fused = _find_op(lr, "fused_mla_attention")
    # 旧 bug 公式给的 cache IO = seqlen*(qk+v)*num_heads*kv_byte; 新公式 = 0.
    # 同时 K/V staging activation (n_blocks_r * seqlen * (qk+v) * num_heads_per_tp * a_byte)
    # 应该出现在 load_act 中, 远大于纯 Q activation IO.
    heads_per_tp = m.num_heads // deploy.tp
    qk_dim = (m.qk_nope_head_dim if m.qk_nope_head_dim > 0 else m.head_dim) + (m.rope_head_dim or 0)
    v_dim = m.v_head_dim if m.v_head_dim > 0 else m.qk_nope_head_dim
    # 下界: 至少包含 Q + K/V staging (n_blocks_r >= 1)
    min_staging = int(128 * (qk_dim + v_dim) * 1 * heads_per_tp * deploy.a_byte)
    assert fused.load_act >= min_staging, (
        f"load_act={fused.load_act} should include K/V staging ≥ {min_staging}")


def test_mla_prefill_chunked_reads_c_kv_cache():
    """MLA chunked prefill 跨 step 时, prior context 应读 c_kv (single-head latent)。"""
    b = _deepseek_v3_bundle(tp_size=8)
    m, deploy, hw = b.model, b.deploy, b.hw
    # 模拟 chunked prefill 第二个 chunk: seq=128 当前 chunk, ctx_len=1024 (其中 896 是 prior)
    lr = dense_layer_time(0, "prefill", tokens=128, ctx_len=1024, model=m, deploy=deploy, hw=hw)
    fused = _find_op(lr, "fused_mla_attention")
    # prior_ctx = 1024 - 128 = 896, c_kv 维 = kv_latent_dim = kv_lora_rank + qk_rope = 512 + 64 = 576
    expected_min = int(896 * 576 * 1 * deploy.kv_byte * 0.5)  # 给点容差
    expected_max = int(896 * 576 * 1 * deploy.kv_byte * 2.0)
    assert expected_min <= fused.load_kv_cache <= expected_max, (
        f"chunked prefill prior cache read should be ~ {896*576*deploy.kv_byte:.0f}, "
        f"got {fused.load_kv_cache}")


def test_mla_prefill_kv_io_far_smaller_than_pre_fix():
    """新公式 KV cache IO 比旧公式 (num_heads * qk+v * kv_byte) 至少小 100×.

    Sanity guard 防回归: 若未来谁误恢复 num_heads 维, 这条立刻挂。
    """
    b = _deepseek_v3_bundle(tp_size=8)
    m, deploy, hw = b.model, b.deploy, b.hw
    seqlen = 128
    lr = dense_layer_time(0, "prefill", tokens=seqlen, ctx_len=seqlen,
                          model=m, deploy=deploy, hw=hw)
    fused = _find_op(lr, "fused_mla_attention")
    heads_per_tp = m.num_heads // deploy.tp  # 128/8 = 16
    qk_dim = (m.qk_nope_head_dim or m.head_dim) + (m.rope_head_dim or 0)  # 192
    v_dim = m.v_head_dim or m.qk_nope_head_dim  # 128
    old_bug_io = seqlen * (qk_dim + v_dim) * heads_per_tp * deploy.kv_byte
    # 新公式 cache IO ≈ 0 (new token); 旧 bug ~ 1.3M+; 实际差距 >> 100×
    assert fused.load_kv_cache * 100 < old_bug_io, (
        f"new cache_io={fused.load_kv_cache} should be << old_bug={old_bug_io} (100×)")


def test_mla_uses_dedicated_op_not_generic_fused_attention():
    """MLA 走专用 `fused_mla_attention` (FlashMLA decode + diff_headdim prefill),
    跟 MHA/GQA 的 `fused_attention` 区分; 防止重构回退把 MLA 误塞回 generic flash 算子.
    """
    b = _deepseek_v3_bundle(tp_size=8)
    lr_dec = dense_layer_time(0, "decode", tokens=1, ctx_len=128,
                              model=b.model, deploy=b.deploy, hw=b.hw)
    assert _has_op(lr_dec, "fused_mla_attention"), "MLA decode op should be fused_mla_attention"
    assert not _has_op(lr_dec, "fused_attention"), "MLA decode must NOT use generic fused_attention"
    lr_pre = dense_layer_time(0, "prefill", tokens=128, ctx_len=128,
                              model=b.model, deploy=b.deploy, hw=b.hw)
    assert _has_op(lr_pre, "fused_mla_attention"), "MLA prefill op should be fused_mla_attention"
    assert not _has_op(lr_pre, "fused_attention"), "MLA prefill must NOT use generic fused_attention"


def test_non_mla_prefill_kv_io_unchanged():
    """非 MLA (GQA) prefill 的 KV cache IO 公式不应受 MLA 修复影响."""
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=4096, num_hidden_layers=36,
        intermediate_size=11008, vocab_size=151936,
        head_dim=128,
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="Qwen/Qwen3-test"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, enable_expert_parallel=False,
        ),
    )
    b = extract_profile_bundle(vc)
    lr = dense_layer_time(0, "prefill", tokens=128, ctx_len=128,
                          model=b.model, deploy=b.deploy, hw=b.hw)
    fused = _find_op(lr, "fused_attention")
    # GQA: kv_io = n_blocks_r * seqlen * head_dim * num_kv_heads * kv_byte * 2
    # n_blocks_r >= 1, 至少应 > 0
    assert fused.load_kv_cache > 0, "GQA prefill should still read KV cache"
