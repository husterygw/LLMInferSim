"""阶段 9.1: DeepSeek-V3.2-Exp DSA cost 公式手算 vs actual 对照。

V3.2 = V3 MLA backbone + DSA lightning indexer + sparse-attended MLA kernel.

跟 V3 path 的差异:
  1. 多 5 个 indexer ops: indexer_wq_b / indexer_wk_weights_proj / indexer_k_norm
     / indexer_q_fp8_quant / sparse_attn_indexer
  2. 主 attention 走 `fused_mla_sparse_attention` (attended_len = min(ctx, index_topk))
     而非 dense `fused_mla_attention`
"""
from types import SimpleNamespace

from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
from llm_infer_sim.core.cost_model.layer_builder import moe_layer_time


def _deepseek_v32_bundle(tp_size: int = 8):
    """DeepSeek-V3.2-Exp 真实 config 字段构造 bundle."""
    hf = SimpleNamespace(
        model_type="deepseek_v32",
        num_attention_heads=128, num_key_value_heads=128,
        hidden_size=7168, num_hidden_layers=61,
        intermediate_size=18432, vocab_size=129280,
        # MLA (跟 V3 同)
        kv_lora_rank=512, q_lora_rank=1536,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        # DSA indexer
        index_head_dim=128, index_n_heads=64, index_topk=2048,
        # MoE
        n_routed_experts=256, num_experts_per_tok=8,
        moe_intermediate_size=2048, n_shared_experts=1, first_k_dense_replace=3,
        # FP8 quant (V3.2 真实)
        quantization_config={"quant_method": "fp8", "activation_scheme": "dynamic"},
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="deepseek-ai/DeepSeek-V3.2-Exp"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, data_parallel_size=1,
            enable_expert_parallel=True,
        ),
        attention_config=SimpleNamespace(backend=None, use_fp4_indexer_cache=False),
    )
    return extract_profile_bundle(vc)


def _has_op(lr, name: str) -> bool:
    return any(op.name == name for op in lr.ops)


def _find_op(lr, name: str):
    for op in lr.ops:
        if op.name == name:
            return op
    raise AssertionError(f"op {name!r} not in {[o.name for o in lr.ops]}")


# ------- Dispatch: V3.2 应该走专用 path -------

def test_v32_uses_sparse_mla_path_not_dense_mla():
    """V3.2 应走 sparse MLA + indexer path, 不应跑 dense MLA `fused_mla_attention`。"""
    b = _deepseek_v32_bundle()
    lr = moe_layer_time(3, "decode", 1, 8192, b.model, b.deploy, b.hw)
    assert _has_op(lr, "fused_mla_sparse_attention"), \
        "V3.2 should use sparse MLA attention"
    assert not _has_op(lr, "fused_mla_attention"), \
        "V3.2 must NOT use dense MLA attention"
    assert not _has_op(lr, "fused_sparse_attention"), \
        "V3.2 must NOT use V4 sparse attention (different layout)"
    assert not _has_op(lr, "fused_attention"), \
        "V3.2 must NOT use generic MHA/GQA fused_attention"


def test_v32_indexer_5_ops_present():
    """V3.2 indexer 子块应当出现 5 个 ops (4 个 GEMM/norm + 1 个 sparse_attn_indexer)。"""
    b = _deepseek_v32_bundle()
    lr = moe_layer_time(3, "decode", 1, 8192, b.model, b.deploy, b.hw)
    for name in ("indexer_wq_b", "indexer_wk_weights_proj", "indexer_k_norm",
                 "indexer_q_fp8_quant", "sparse_attn_indexer"):
        assert _has_op(lr, name), f"V3.2 indexer op {name} missing"


def test_v32_mla_backbone_ops_still_present():
    """V3.2 仍然走 MLA Q/KV proj backbone (跟 V3 一致)."""
    b = _deepseek_v32_bundle()
    lr = moe_layer_time(3, "decode", 1, 8192, b.model, b.deploy, b.hw)
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        assert _has_op(lr, name), f"V3.2 MLA backbone op {name} missing"


# ------- Indexer wq_b 手算 -------

def test_indexer_wq_b_handcheck():
    """indexer_wq_b: q_lora_rank=1536 → n_head*head_dim = 64*128 = 8192.
    FLOPs = 2 × tokens × 1536 × 8192.
    """
    b = _deepseek_v32_bundle(tp_size=8)
    lr = moe_layer_time(3, "decode", 1, 4096, b.model, b.deploy, b.hw)
    op = _find_op(lr, "indexer_wq_b")
    tokens = 1
    expected_flops = 2 * tokens * 1536 * (64 * 128)
    assert op.flops == expected_flops, \
        f"indexer_wq_b flops={op.flops} expected {expected_flops}"


def test_indexer_wk_weights_proj_handcheck():
    """indexer_wk_weights_proj: hidden=7168 → (head_dim + n_head) = 128+64 = 192.
    FLOPs = 2 × tokens × 7168 × 192.  bf16 weight (quant_config=None).
    """
    b = _deepseek_v32_bundle()
    lr = moe_layer_time(3, "decode", 1, 4096, b.model, b.deploy, b.hw)
    op = _find_op(lr, "indexer_wk_weights_proj")
    tokens = 1
    expected_flops = 2 * tokens * 7168 * (128 + 64)
    assert op.flops == expected_flops
    assert op.op_precision == "bf16"   # quant_config=None
    # bf16 weight bytes: hidden * (head_dim + n_head) * 2.0
    assert op.load_weight == 7168 * 192 * 2


# ------- Sparse attention attended_len 限制 -------

def test_sparse_attended_len_capped_at_index_topk():
    """attended_len = min(ctx_len, index_topk=2048).

    ctx=1024 (<topk): attended=1024 (degenerate to dense)
    ctx=8192 (>topk): attended=2048 (sparse 起效)
    """
    b = _deepseek_v32_bundle()
    # ctx=1024 < topk=2048 → attended=1024
    lr_short = moe_layer_time(3, "decode", 1, 1024, b.model, b.deploy, b.hw)
    op_short = _find_op(lr_short, "fused_mla_sparse_attention")
    # ctx=8192 > topk=2048 → attended=2048
    lr_long = moe_layer_time(3, "decode", 1, 8192, b.model, b.deploy, b.hw)
    op_long = _find_op(lr_long, "fused_mla_sparse_attention")
    # 8x ctx 增长, 但 attended 被 topk 截断 → flops 不应该按比例增, 而应该 ~2x
    # (1024 → 2048, 而非 1024 → 8192)
    ratio = op_long.flops / op_short.flops
    assert 1.8 < ratio < 2.2, \
        f"sparse attended ratio for ctx 8192/1024 should be ~2 (capped at topk), got {ratio:.2f}"


def test_v32_vs_v3_dense_ratio_at_long_ctx():
    """长 ctx 下 V3.2 attention FLOPs 应远小于 V3 dense MLA (受 index_topk 限制)。"""
    b = _deepseek_v32_bundle()
    # 模拟同 backbone 但跑 dense (临时改 index_topk=0 触发 dense MLA path)
    from llm_infer_sim.core.profiles.model_config import ModelConfig
    from dataclasses import replace
    m_dense = replace(b.model, index_topk=0)
    lr_sparse = moe_layer_time(3, "decode", 1, 16384, b.model, b.deploy, b.hw)
    lr_dense = moe_layer_time(3, "decode", 1, 16384, m_dense, b.deploy, b.hw)
    sparse_attn = _find_op(lr_sparse, "fused_mla_sparse_attention")
    dense_attn = _find_op(lr_dense, "fused_mla_attention")
    # 16384/2048 = 8x, V3.2 sparse 主 attention 应当至少便宜 5×
    assert dense_attn.flops > sparse_attn.flops * 5, \
        f"long ctx V3.2 sparse should be much cheaper than dense MLA: " \
        f"sparse={sparse_attn.flops:,} dense={dense_attn.flops:,}"


# ------- sparse_attn_indexer KV cache IO -------

def test_sparse_attn_indexer_kv_cache_io_scales_with_ctx():
    """sparse_attn_indexer 读 indexer KV cache, IO 应跟 ctx_len 成正比 (单 head)."""
    b = _deepseek_v32_bundle()
    lr_1k = moe_layer_time(3, "decode", 1, 1024, b.model, b.deploy, b.hw)
    lr_8k = moe_layer_time(3, "decode", 1, 8192, b.model, b.deploy, b.hw)
    op_1k = _find_op(lr_1k, "sparse_attn_indexer")
    op_8k = _find_op(lr_8k, "sparse_attn_indexer")
    ratio = op_8k.load_kv_cache / op_1k.load_kv_cache
    assert 7.5 < ratio < 8.5, \
        f"indexer KV IO should scale linearly with ctx (8192/1024 ≈ 8), got {ratio:.2f}"
