"""阶段 9.1: DeepSeek-V3.2-Exp DSA cost 公式手算 vs actual (新架构).

V3.2 = V3 MLA backbone + DSA lightning indexer + sparse-attended MLA kernel.

迁移自旧 cost_model.layer_builder._build_v32_mla_sparse_attention_block. 现在跑的是
DeepSeekModelTemplate (auto-detect index_topk > 0 走 V3.2 sparse path).
"""
from __future__ import annotations

from llm_infer_sim.core.cost.engine import build_deepseek_roofline_engine
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


def _deepseek_v32(*, tp: int = 8, index_topk: int = 2048):
    return ModelConfig(
        name="DeepSeek-V3.2-Exp",
        hidden_dim=7168, num_heads=128, num_kv_heads=128, head_dim=56,
        ffn_dim=18432, num_layers=61, vocab_size=129280,
        is_moe=True, num_experts=256, num_activated_experts=8,
        expert_dim=2048, num_shared_experts=1,
        moe_layer_freq=1, first_moe_layer=3,
        kv_lora_rank=512, kv_latent_dim=576,
        qk_nope_head_dim=128, v_head_dim=128, rope_head_dim=64,
        q_lora_rank=1536,
        index_n_heads=64, index_head_dim=128, index_topk=index_topk,
    ), DeployConfig(tp_size=tp)


def _attn_ops(model, deploy, *, tokens=1, phase="decode", ctx_len=8192, layer_idx=3):
    """Build MLA attention block ops for one layer (V3.2 sparse or V3 dense based on model)."""
    hw = get_hardware_profile("RTX_4090")
    engine = build_deepseek_roofline_engine(model, deploy, hw)
    if phase == "decode":
        wl = GlobalStepWorkload(
            step_id=0, phase=StepPhase.DECODE,
            requests=[RequestWorkload(
                request_id="d", phase=StepPhase.DECODE,
                num_tokens=1, context_len=ctx_len,
            )] * tokens,
            num_decode_tokens=tokens, total_scheduled_tokens=tokens,
            num_decode_requests=tokens,
        )
    else:
        wl = GlobalStepWorkload(
            step_id=0, phase=StepPhase.PREFILL,
            requests=[RequestWorkload(
                request_id="r", phase=StepPhase.PREFILL,
                num_tokens=tokens, context_len=ctx_len - tokens,
            )],
            num_prefill_tokens=tokens, total_scheduled_tokens=tokens,
            num_prefill_requests=1,
        )
    step = StepShape.from_workload(wl, deploy)
    return engine.template._build_mla_attn_block(layer_idx, step, engine.factories)


def _has_subtype(ops, subtype) -> bool:
    return any(op.op_subtype == subtype for op in ops)


def _find_by_subtype(ops, subtype):
    for op in ops:
        if op.op_subtype == subtype:
            return op
    raise AssertionError(f"op_subtype {subtype!r} not in {[o.op_subtype for o in ops]}")


def _find_by_name_endswith(ops, suffix):
    for op in ops:
        if op.name.endswith(suffix):
            return op
    raise AssertionError(f"no op ends with {suffix!r} in {[o.name for o in ops]}")


# ---- V3.2 应走 sparse MLA path ----

def test_v32_uses_sparse_mla_path_not_dense_mla():
    """index_topk > 0 → V3.2 sparse MLA + indexer block, 不走 dense MLA."""
    m, deploy = _deepseek_v32()
    ops = _attn_ops(m, deploy, ctx_len=8192)
    attn = next(o for o in ops if o.op_kind == "attention" and "sparse" in (o.tags or ()))
    assert attn is not None
    assert attn.name.endswith("_mla_sparse_attention")
    # 没有 dense MLA op (tag mla 但不含 sparse)
    dense_mla = [o for o in ops if o.op_kind == "attention"
                 and "mla" in (o.tags or ())
                 and "sparse" not in (o.tags or ())
                 and o.op_subtype in ("prefill", "decode")
                 and "indexer" not in o.name]
    assert not dense_mla


def test_v32_indexer_5_ops_present():
    """V3.2 indexer 子块: 5 个 ops."""
    m, deploy = _deepseek_v32()
    ops = _attn_ops(m, deploy)
    for subtype in ("indexer_wq_b", "indexer_wk_weights_proj",
                    "rmsnorm",   # indexer_k_norm
                    "quantize",  # indexer_q_fp8_quant
                    "sparse_index"):
        assert _has_subtype(ops, subtype), f"missing {subtype}"
    # 5 ops 全部 by name 也找得到
    for name_suffix in ("_indexer_wq_b", "_indexer_wk_weights_proj",
                        "_indexer_k_norm", "_indexer_q_fp8_quant",
                        "_sparse_attn_indexer"):
        _find_by_name_endswith(ops, name_suffix)


def test_v32_mla_backbone_ops_still_present():
    """V3.2 仍然走 MLA Q/KV proj backbone."""
    m, deploy = _deepseek_v32()
    ops = _attn_ops(m, deploy)
    for subtype in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        assert _has_subtype(ops, subtype)


# ---- 手算 ----

def test_indexer_wq_b_handcheck():
    """indexer_wq_b: q_lora_rank=1536 → n_head*head_dim = 64*128 = 8192.
    FLOPs = 2 × tokens × 1536 × 8192."""
    m, deploy = _deepseek_v32()
    ops = _attn_ops(m, deploy, ctx_len=4096)
    op = _find_by_subtype(ops, "indexer_wq_b")
    tokens = 1
    expected = 2 * tokens * m.q_lora_rank * (m.index_n_heads * m.index_head_dim)
    assert op.formula().flops == expected


def test_indexer_wk_weights_proj_handcheck():
    """indexer_wk_weights_proj: hidden=7168 → (head_dim + n_head) = 192. bf16."""
    m, deploy = _deepseek_v32()
    ops = _attn_ops(m, deploy, ctx_len=4096)
    op = _find_by_subtype(ops, "indexer_wk_weights_proj")
    tokens = 1
    fused_oc = m.index_head_dim + m.index_n_heads
    f = op.formula()
    assert f.flops == 2 * tokens * m.hidden_dim * fused_oc
    assert f.op_precision == "bf16"
    assert f.load_weight == m.hidden_dim * fused_oc * 2   # bf16 weight


# ---- sparse attended_len 截断 ----

def test_sparse_attended_len_capped_at_index_topk():
    """attended_len = min(ctx_len, index_topk=2048).
    ctx=1024 → attended=1024, ctx=8192 → attended=2048. flops 比 ~ 2x (不是 8x)."""
    m, deploy = _deepseek_v32()
    ops_short = _attn_ops(m, deploy, ctx_len=1024)
    ops_long = _attn_ops(m, deploy, ctx_len=8192)
    attn_short = _find_by_name_endswith(ops_short, "_mla_sparse_attention")
    attn_long = _find_by_name_endswith(ops_long, "_mla_sparse_attention")
    ratio = attn_long.formula().flops / attn_short.formula().flops
    assert 1.8 < ratio < 2.2, f"ratio for 8192/1024 should be ~2 (capped at topk), got {ratio:.2f}"


def test_v32_vs_v3_dense_ratio_at_long_ctx():
    """长 ctx 下 V3.2 sparse 比 V3 dense MLA 便宜 ≥5×."""
    m_sparse, deploy = _deepseek_v32(index_topk=2048)
    # 临时 index_topk=0 → V3 dense path
    from dataclasses import replace
    m_dense = replace(m_sparse, index_topk=0)
    ops_sparse = _attn_ops(m_sparse, deploy, ctx_len=16384)
    ops_dense = _attn_ops(m_dense, deploy, ctx_len=16384)
    attn_sparse = _find_by_name_endswith(ops_sparse, "_mla_sparse_attention")
    attn_dense = _find_by_name_endswith(ops_dense, "_mla_attention")
    assert attn_dense.formula().flops > attn_sparse.formula().flops * 5


# ---- sparse_attn_indexer KV cache IO scales with ctx ----

def test_sparse_attn_indexer_kv_cache_io_scales_with_ctx():
    """sparse_attn_indexer KV IO 跟 ctx_len 线性 (8192/1024 ≈ 8)."""
    m, deploy = _deepseek_v32()
    ops_1k = _attn_ops(m, deploy, ctx_len=1024)
    ops_8k = _attn_ops(m, deploy, ctx_len=8192)
    op_1k = _find_by_subtype(ops_1k, "sparse_index")
    op_8k = _find_by_subtype(ops_8k, "sparse_index")
    ratio = op_8k.formula().load_kv_cache / op_1k.formula().load_kv_cache
    assert 7.5 < ratio < 8.5, f"ratio={ratio:.2f}"
