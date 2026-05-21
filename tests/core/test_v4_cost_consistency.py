"""阶段 9-δ: DeepSeek-V4 cost 公式手算 vs actual (新架构).

迁移自旧 cost_model.layer_builder._build_v4_sparse_attention_block + _build_v4_compressor_ops.
现在跑 DeepSeekModelTemplate (auto-detect window_size>0 AND o_groups>0 走 V4 path).

覆盖 V4-Flash:
  - fused_wqa_wkv / fused_compress_wkv_wgate / fused_index_compress_wkv_wgate
  - index_wq_b / index_weights_proj 不切 TP
  - wo_a / wo_b 用 deploy.w_byte
  - hash MoE routing (前 num_hash_layers 层无 moe_gate)
  - HC pre/post + model-level hc_embedding_repeat / hc_head / final_norm
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.engine import build_deepseek_roofline_engine
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


W_BYTE = 2.0
A_BYTE = 2.0


def _v4_flash(*, tp: int = 8, indexer_kv_byte: float = 1.0) -> tuple[ModelConfig, DeployConfig, float]:
    """V4-Flash hf_config (真实) → ModelConfig + DeployConfig."""
    cratios = [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
               4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
               4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
    m = ModelConfig(
        name="DeepSeek-V4-Flash",
        hidden_dim=4096, num_heads=64, num_kv_heads=1, head_dim=512,
        ffn_dim=0, num_layers=43, vocab_size=129280,
        is_moe=True, num_experts=256, num_activated_experts=6,
        expert_dim=2048, num_shared_experts=1,
        moe_layer_freq=1, first_moe_layer=0,
        q_lora_rank=1024, o_lora_rank=1024, o_groups=8,
        rope_head_dim=64,
        window_size=128, compress_ratios=cratios,
        index_topk=512, index_n_heads=64, index_head_dim=128,
        hc_mult=4, hc_sinkhorn_iters=20,
        expert_fp4=True, num_hash_layers=3,
    )
    return m, DeployConfig(tp_size=tp), indexer_kv_byte


def _build_engine(*, tp: int = 8, indexer_kv_byte: float = 1.0):
    m, deploy, ikb = _v4_flash(tp=tp, indexer_kv_byte=indexer_kv_byte)
    hw = get_hardware_profile("RTX_4090")
    engine = build_deepseek_roofline_engine(m, deploy, hw, indexer_kv_byte=ikb)
    return m, deploy, engine


def _decode_step(deploy, *, tokens=1, ctx=128):
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.DECODE,
        requests=[RequestWorkload(
            request_id="d", phase=StepPhase.DECODE,
            num_tokens=1, context_len=ctx,
        )] * tokens,
        num_decode_tokens=tokens, total_scheduled_tokens=tokens,
        num_decode_requests=tokens,
    )
    return StepShape.from_workload(wl, deploy)


def _layer_ops(layer_idx: int, *, tp=8, indexer_kv_byte=1.0):
    """Build full layer ops (attn block + ffn block) for V4."""
    m, deploy, engine = _build_engine(tp=tp, indexer_kv_byte=indexer_kv_byte)
    step = _decode_step(deploy)
    attn = engine.template._build_v4_attn_block(layer_idx, step, engine.factories)
    ffn = engine.template._build_moe_ffn_block(layer_idx, step, engine.factories)
    return m, deploy, attn + ffn


def _find_by_name_endswith(ops, suffix):
    for op in ops:
        if op.name.endswith(suffix):
            return op
    raise AssertionError(f"no op ends with {suffix!r} in {[o.name for o in ops]}")


def _has_op_endswith(ops, suffix) -> bool:
    return any(op.name.endswith(suffix) for op in ops)


def _find_by_subtype(ops, subtype):
    for op in ops:
        if op.op_subtype == subtype:
            return op
    raise AssertionError(f"op_subtype {subtype!r} not in {[o.op_subtype for o in ops]}")


def _has_subtype(ops, subtype) -> bool:
    return any(op.op_subtype == subtype for op in ops)


# ============================================================================
# V4 字段透传 + is_v4 激活
# ============================================================================

def test_v4_fields_all_passthrough():
    m, _, _ = _build_engine()
    assert m.window_size == 128
    assert m.o_groups == 8
    assert m.o_lora_rank == 1024
    assert m.q_lora_rank == 1024
    assert m.index_topk == 512
    assert m.index_n_heads == 64
    assert m.index_head_dim == 128
    assert m.hc_mult == 4
    assert m.hc_sinkhorn_iters == 20
    assert m.expert_fp4 is True
    assert len(m.compress_ratios) == 44
    assert m.compress_ratios[:6] == [0, 0, 4, 128, 4, 128]
    assert m.rope_head_dim == 64


def test_v4_is_v4_path_activated():
    m, deploy, engine = _build_engine()
    step = _decode_step(deploy)
    plan = engine.template.build_step(step, engine.factories)
    assert plan.metadata["is_v4"] is True


def test_v4_compress_ratio_routing():
    m, _, _ = _build_engine()
    assert m.get_compress_ratio(0) == 0
    assert m.get_compress_ratio(1) == 0
    assert m.get_compress_ratio(2) == 4
    assert m.get_compress_ratio(3) == 128
    assert m.get_compress_ratio(4) == 4
    assert m.get_compress_ratio(42) == 4


def test_v4_no_dense_ffn_intermediate_size():
    m, _, _ = _build_engine()
    assert m.ffn_dim == 0
    assert m.is_moe
    assert m.first_moe_layer == 0


# ============================================================================
# fused_wqa_wkv 手算
# ============================================================================

def test_fused_wqa_wkv_handcheck():
    """flops = 2 × tokens × h × (q_lora + head_dim). load_act = h × tokens × a_byte (x 一次)."""
    m, _, ops = _layer_ops(0)
    op = _find_by_subtype(ops, "fused_wqa_wkv")
    tokens = 1
    expected_flops = 2 * tokens * m.hidden_dim * (m.q_lora_rank + m.head_dim)
    f = op.formula()
    assert f.flops == expected_flops    # 12,582,912
    assert f.load_act == m.hidden_dim * tokens * A_BYTE
    # 防御: 旧拆分代码 load_act = 2 * h * tokens
    assert f.load_act != 2 * m.hidden_dim * tokens * A_BYTE


def test_wq_a_and_wkv_no_longer_present():
    _, _, ops = _layer_ops(0)
    assert _has_subtype(ops, "fused_wqa_wkv")
    assert not _has_subtype(ops, "wq_a")
    assert not _has_subtype(ops, "wkv")


# ============================================================================
# fused_compress_wkv_wgate 手算
# ============================================================================

def test_fused_compress_wkv_wgate_handcheck_ratio4():
    """ratio=4 → coff=2 → fused_oc = 2*2*head_dim = 2048."""
    m, _, ops = _layer_ops(2)   # layer 2 ratio=4
    op = _find_by_subtype(ops, "fused_compress_wkv_wgate")
    fused_oc = 2 * 2 * m.head_dim
    assert op.formula().flops == 2 * 1 * m.hidden_dim * fused_oc


def test_fused_compress_wkv_wgate_handcheck_ratio128():
    """ratio=128 → coff=1 → fused_oc = 2*1*head_dim = 1024."""
    m, _, ops = _layer_ops(3)   # layer 3 ratio=128
    op = _find_by_subtype(ops, "fused_compress_wkv_wgate")
    fused_oc = 2 * 1 * m.head_dim
    assert op.formula().flops == 2 * 1 * m.hidden_dim * fused_oc


def test_compress_wkv_and_gate_no_longer_present():
    _, _, ops = _layer_ops(2)
    assert _has_subtype(ops, "fused_compress_wkv_wgate")
    assert not _has_subtype(ops, "compress_wkv")
    assert not _has_subtype(ops, "compress_gate")


# ============================================================================
# Indexer 不切 TP
# ============================================================================

def test_index_wq_b_does_not_divide_tp():
    """index_wq_b ReplicatedLinear: 用完整 index_n_heads 而非 /tp."""
    m, deploy, ops = _layer_ops(2)
    op = _find_by_subtype(ops, "index_wq_b")
    expected = 2 * 1 * m.q_lora_rank * m.index_n_heads * m.index_head_dim
    assert op.formula().flops == expected
    assert op.formula().flops != expected // deploy.tp_size


def test_index_weights_proj_does_not_divide_tp():
    m, deploy, ops = _layer_ops(2)
    op = _find_by_subtype(ops, "index_weights_proj")
    expected = 2 * 1 * m.hidden_dim * m.index_n_heads
    assert op.formula().flops == expected
    assert op.formula().flops != expected // deploy.tp_size


# ============================================================================
# index_score 用 indexer_kv_byte (fp8=1.0 / fp4=0.5)
# ============================================================================

def test_index_score_kv_cache_byte_matches_deploy():
    """切到 fp4 indexer cache, K cache 字节减半."""
    _, _, ops_fp8 = _layer_ops(2, indexer_kv_byte=1.0)
    _, _, ops_fp4 = _layer_ops(2, indexer_kv_byte=0.5)
    op_fp8 = _find_by_subtype(ops_fp8, "index_score")
    op_fp4 = _find_by_subtype(ops_fp4, "index_score")
    assert op_fp4.formula().load_kv_cache == pytest.approx(
        op_fp8.formula().load_kv_cache / 2,
    )


# ============================================================================
# wo_a 用 w_byte 而非 hardcoded
# ============================================================================

def test_wo_a_uses_deploy_w_byte_not_hardcoded():
    m, deploy, ops = _layer_ops(0)
    op = _find_by_subtype(ops, "wo_a")
    n_local_groups = max(m.o_groups // deploy.tp_size, 1)
    heads_per_tp = m.num_heads // deploy.tp_size
    wo_a_ic = heads_per_tp * m.head_dim // n_local_groups
    wo_a_oc = n_local_groups * m.o_lora_rank
    expected_weight = int(wo_a_ic * wo_a_oc * W_BYTE)
    assert op.formula().load_weight == expected_weight


def test_v3_mla_wkv_separate_not_in_v4_path():
    """V4 path 用 fused_wqa_wkv, 不应再出现 wkv (MLA 字段)."""
    _, _, ops = _layer_ops(0)
    assert not _has_subtype(ops, "wkv")


# ============================================================================
# 完整 layer 2 路径
# ============================================================================

def test_v4_layer_2_full_path_ops_present():
    """layer 2 (ratio=4 CSA + num_hash_layers=3) 完整 V4 ops (by op_subtype or name suffix)."""
    _, _, ops = _layer_ops(2)
    # by op_subtype (clean semantic identifiers)
    for sub in (
        "fused_wqa_wkv",
        "wq_b",
        "fused_compress_wkv_wgate",
        "compress_pool",
        "fused_index_compress_wkv_wgate",
        "index_wq_b", "index_weights_proj", "index_score",
        "wo_a", "wo_b",
        "moe_hash_lookup",
        "fused_moe",   # routed_experts
        "shared_expert_up_gate", "shared_expert_down",
    ):
        assert _has_subtype(ops, sub), f"layer 2 missing op_subtype={sub}"
    # q_norm / kv_norm 都是 op_kind=norm op_subtype=rmsnorm, 用 name 区分
    assert _has_op_endswith(ops, "_q_norm")
    assert _has_op_endswith(ops, "_kv_norm")
    # 必须有 hc_attn_pre/post + hc_ffn_pre/post (V4 hc_mult=4)
    assert _has_op_endswith(ops, "hc_attn_pre")
    assert _has_op_endswith(ops, "hc_attn_post")
    assert _has_op_endswith(ops, "hc_ffn_pre")
    assert _has_op_endswith(ops, "hc_ffn_post")
    # attn_allreduce 因 tp=8 > 1 存在
    assert _has_op_endswith(ops, "attn_allreduce")
    # layer 2 是 hash 层, 没 moe_gate
    assert not _has_subtype(ops, "router")     # moe_gate 的 op_subtype


# ============================================================================
# V4 hash MoE routing
# ============================================================================

def test_v4_hash_layers_use_lookup_not_gate():
    """num_hash_layers=3 → layer 0/1/2 走 lookup, 没 moe_gate."""
    for layer_idx in (0, 1, 2):
        _, _, ops = _layer_ops(layer_idx)
        assert _has_subtype(ops, "moe_hash_lookup")
        assert not _has_subtype(ops, "router")    # moe_gate.op_subtype = "router"


def test_v4_non_hash_layers_still_use_gate():
    """layer >= 3 应该走正常 moe_gate."""
    for layer_idx in (3, 4, 5, 42):
        _, _, ops = _layer_ops(layer_idx)
        assert _has_subtype(ops, "router"), f"layer {layer_idx} missing moe_gate"
        assert not _has_subtype(ops, "moe_hash_lookup"), f"layer {layer_idx} has hash lookup"


def test_v4_hash_lookup_zero_flops():
    """hash lookup FLOPs = 0 (查表, 不是 GEMM)."""
    _, _, ops = _layer_ops(0)
    op = _find_by_subtype(ops, "moe_hash_lookup")
    assert op.formula().flops == 0


# ============================================================================
# HC model-level ops (hc_embedding_repeat / hc_head / final_norm)
# ============================================================================

def test_v4_model_core_has_hc_head_and_final_norm():
    """V4 plan 含 hc_embedding_repeat + hc_head + final_norm."""
    _, deploy, engine = _build_engine()
    step = _decode_step(deploy)
    plan = engine.template.build_step(step, engine.factories)
    names = {op.name for op in plan.ops}
    assert "hc_embedding_repeat" in names
    assert "hc_head" in names
    assert "final_norm" in names


def test_non_hc_model_no_hc_model_level_ops():
    """Qwen3-30B-A3B hc_mult=0, plan 不该有 HC model-level ops."""
    qwen = ModelConfig(
        name="Qwen3-30B-A3B",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=0, num_layers=48, vocab_size=151936,
        is_moe=True, num_experts=128, num_activated_experts=8,
        expert_dim=768, num_shared_experts=0,
        moe_layer_freq=1, first_moe_layer=0,
    )
    from llm_infer_sim.core.cost.engine import build_qwen_roofline_engine
    e = build_qwen_roofline_engine(qwen, DeployConfig(tp_size=2), get_hardware_profile("RTX_4090"))
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.DECODE,
        requests=[RequestWorkload(
            request_id="d", phase=StepPhase.DECODE, num_tokens=1, context_len=128,
        )],
        num_decode_tokens=1, total_scheduled_tokens=1, num_decode_requests=1,
    )
    step = StepShape.from_workload(wl, e.deploy)
    plan = e.template.build_step(step, e.factories)
    names = {op.name for op in plan.ops}
    assert "hc_embedding_repeat" not in names
    assert "hc_head" not in names
    assert "final_norm" not in names


# ============================================================================
# attn_sink: sparse attention 每 query +1 attended pos
# ============================================================================

def test_attn_sink_increments_attended_count():
    """sparse decode attention: attended = window + compressed + 1 (sink).
    手算: window=64, ctx=128, compress_ratio=4, index_topk=16 →
        all_compressed = 128/4 = 32, compressed_attended = min(16, 32) = 16,
        attended = min(128, 64) + 16 + 1 = 81."""
    from llm_infer_sim.core.operators.factories.v4_attention import V4AttentionOpFactory
    m = ModelConfig(
        name="x", hidden_dim=1024, num_heads=8, num_kv_heads=1, head_dim=512,
        ffn_dim=0, num_layers=1, vocab_size=1000,
        window_size=64, o_groups=1, o_lora_rank=128,
        index_topk=16, index_n_heads=8, index_head_dim=128,
    )
    deploy = DeployConfig(tp_size=1)
    hw = get_hardware_profile("RTX_4090")
    factory = V4AttentionOpFactory(m, deploy, hw, a_byte=2.0, kv_byte=1.0)
    step = _decode_step(deploy, tokens=1, ctx=128)
    op = factory.sparse_attention(0, step, compress_ratio=4)
    # attended = 64 + 16 + 1 = 81
    expected_qk = 81 * m.head_dim * 8 * 1 * 2
    expected_sv = 1 * m.head_dim * 81 * 8 * 1 * 2
    expected_softmax = 1 * 8 * 81 * 1 * 5
    assert op.formula().flops == expected_qk + expected_sv + expected_softmax


# ============================================================================
# layer 0 (SWA-only) / layer 3 (HCA): 局部 ops 存在性
# ============================================================================

def test_v4_layer_0_swa_only_no_compressor_no_indexer():
    """layer 0 ratio=0: 没 compressor / indexer, 但 fused_wqa_wkv + sparse_attention 在."""
    _, _, ops = _layer_ops(0)
    assert _has_subtype(ops, "fused_wqa_wkv")
    # sparse_attention op_subtype = "decode" or "prefill"; 用 name 后缀更稳
    assert _has_op_endswith(ops, "_fused_sparse_attention")
    assert not _has_subtype(ops, "fused_compress_wkv_wgate")
    assert not _has_subtype(ops, "fused_index_compress_wkv_wgate")
    assert not _has_subtype(ops, "index_score")


def test_v4_layer_3_hca_compressor_but_no_indexer():
    """layer 3 ratio=128 HCA: 有 compressor, 但**没有** indexer."""
    _, _, ops = _layer_ops(3)
    assert _has_subtype(ops, "fused_compress_wkv_wgate")
    assert _has_subtype(ops, "compress_pool")
    assert not _has_subtype(ops, "fused_index_compress_wkv_wgate")
    assert not _has_subtype(ops, "index_score")
