"""MLA cost 公式手算 vs actual 对照 (V3 §4.1.4 + DeepSeekModel).

链路: DeepSeekModel._build_mla_attn_block 直接构造 GEMM (q/kv proj) + Attention (MLA).

覆盖 5 个 MLA 核心 op 公式 (DeepSeek-V3 tp=8 decode tokens=1):
  1. q_a_proj:           h → q_lora_rank
  2. q_b_proj:           q_lora_rank → heads_per_tp × (qk_nope + qk_rope)
  3. kv_a_proj_with_mqa: h → kv_lora_rank + qk_rope  (output to KV cache)
  4. kv_b_proj:          kv_lora_rank → heads_per_tp × (qk_nope + v_head_dim)
  5. o_proj:             heads_per_tp × v_head_dim → h
"""
from __future__ import annotations

from llm_infer_sim.core.cost.engine import build_deepseek_roofline_engine
from llm_infer_sim.core.step.step_shape import StepShape
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from llm_infer_sim.core.models.config import ModelConfig
from tests.helpers.support import make_model_config
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)

W_BYTE = 2.0
A_BYTE = 2.0


def _deepseek_v3(tp: int = 8) -> tuple[ModelConfig, DeploymentProfile, RuntimeProfile]:
    model = make_model_config(
        name="DeepSeek-V3",
        # head_dim=56 用旧测试同样的 hidden/num_heads (7168/128); MLA 不真的用 head_dim
        hidden_dim=7168, num_heads=128, num_kv_heads=128, head_dim=56,
        ffn_dim=18432, num_layers=61, vocab_size=129280,
        is_moe=True, num_experts=256, num_activated_experts=8,
        expert_dim=2048, num_shared_experts=1,
        moe_layer_freq=1, first_moe_layer=3,
        kv_lora_rank=512, kv_latent_dim=576,
        qk_nope_head_dim=128, v_head_dim=128, rope_head_dim=64,
        q_lora_rank=1536,
    )
    deployment = DeploymentProfile.flat(tp=tp)
    runtime = RuntimeProfile.flat()
    return model, deployment, runtime


def _v3_lite_no_q_lora(tp: int = 1) -> tuple[ModelConfig, DeploymentProfile, RuntimeProfile]:
    """V2-Lite 类: kv_lora_rank > 0 但 q_lora_rank == 0."""
    model = make_model_config(
        name="DeepSeek-V2-Lite-like",
        hidden_dim=2048, num_heads=16, num_kv_heads=16, head_dim=128,
        ffn_dim=10944, num_layers=27, vocab_size=102400,
        is_moe=True, num_experts=64, num_activated_experts=6,
        expert_dim=1408, num_shared_experts=2,
        moe_layer_freq=1, first_moe_layer=1,
        kv_lora_rank=512, kv_latent_dim=576,
        qk_nope_head_dim=128, v_head_dim=128, rope_head_dim=64,
        q_lora_rank=0,
    )
    deployment = DeploymentProfile.flat(tp=tp)
    runtime = RuntimeProfile.flat()
    return model, deployment, runtime


def _layer_ops(model: ModelConfig, deployment: DeploymentProfile,
               runtime: RuntimeProfile, *,
               tokens: int, phase: str, layer_idx: int = 0, ctx: int = 128):
    """构造 layer ops list (只一层, 避免多层重名干扰)."""
    hw = get_hardware_profile("RTX_4090")
    engine = build_deepseek_roofline_engine(model, deployment, runtime, hw)

    if phase == "prefill":
        wl = GlobalStepWorkload(
            step_id=0, phase=StepPhase.PREFILL,
            requests=[RequestWorkload(
                request_id="r0", phase=StepPhase.PREFILL,
                num_tokens=tokens, context_len=0,
            )],
            num_prefill_tokens=tokens, total_scheduled_tokens=tokens,
            num_prefill_requests=1,
        )
    else:
        wl = GlobalStepWorkload(
            step_id=0, phase=StepPhase.DECODE,
            requests=[RequestWorkload(
                request_id="d", phase=StepPhase.DECODE,
                num_tokens=1, context_len=ctx,
            )] * tokens,
            num_decode_tokens=tokens, total_scheduled_tokens=tokens,
            num_decode_requests=tokens,
        )
    step = StepShape.from_workload(wl, "eager")
    return engine.model._build_mla_attn_block(layer_idx, step)


def _chunked_prefill_layer_ops(model, deployment, runtime, *, current_tokens, total_ctx, layer_idx=0):
    """模拟 chunked prefill: 当前 chunk current_tokens, 之前已有 prior = total_ctx - current_tokens."""
    hw = get_hardware_profile("RTX_4090")
    engine = build_deepseek_roofline_engine(model, deployment, runtime, hw)
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=current_tokens, context_len=total_ctx - current_tokens,
        )],
        num_prefill_tokens=current_tokens, total_scheduled_tokens=current_tokens,
        num_prefill_requests=1,
    )
    step = StepShape.from_workload(wl, "eager")
    return engine.model._build_mla_attn_block(layer_idx, step)


def _find_by_subtype(ops, subtype):
    for op in ops:
        if op.op_subtype == subtype:
            return op
    raise AssertionError(f"op_subtype {subtype!r} not in {[o.op_subtype for o in ops]}")


def _has_subtype(ops, subtype) -> bool:
    return any(op.op_subtype == subtype for op in ops)


# ---- MLA structural checks ----

def test_mla_layer_uses_q_a_q_b_not_single_q_proj():
    """V3 (q_lora_rank > 0) 必须用 q_a_proj + q_b_proj, 不是单 q_proj 也不是 fused qkv."""
    m, deployment, runtime = _deepseek_v3()
    ops = _layer_ops(m, deployment, runtime, tokens=1, phase="decode")
    assert _has_subtype(ops, "q_a_proj")
    assert _has_subtype(ops, "q_b_proj")
    assert not _has_subtype(ops, "q_proj")
    assert not _has_subtype(ops, "qkv_proj")


def test_mla_layer_no_separate_k_v_proj():
    """MLA 不应有独立的 k_proj/v_proj, 而是 kv_a_proj_with_mqa (single fused)."""
    m, deployment, runtime = _deepseek_v3()
    ops = _layer_ops(m, deployment, runtime, tokens=1, phase="decode")
    assert _has_subtype(ops, "kv_a_proj_with_mqa")
    assert not _has_subtype(ops, "k_proj")
    assert not _has_subtype(ops, "v_proj")


# ---- 5 个 MLA proj 手算 ----

def test_q_a_proj_handcheck():
    """q_a_proj: h → q_lora_rank, flops = 2 × tokens × h × q_lora_rank."""
    m, deployment, runtime = _deepseek_v3()
    tokens = 1
    ops = _layer_ops(m, deployment, runtime, tokens=tokens, phase="decode")
    op = _find_by_subtype(ops, "q_a_proj")
    expected = 2 * tokens * m.hidden_dim * m.q_lora_rank
    assert op.roofline_spec().flops == expected   # 2*1*7168*1536 = 22,020,096


def test_q_b_proj_handcheck():
    """q_b_proj: q_lora_rank → heads_per_tp × (qk_nope + qk_rope)."""
    m, deployment, runtime = _deepseek_v3()
    tokens = 1
    ops = _layer_ops(m, deployment, runtime, tokens=tokens, phase="decode")
    op = _find_by_subtype(ops, "q_b_proj")
    heads_per_tp = m.num_heads // deployment.parallelism.tp
    q_head_dim = m.qk_nope_head_dim + m.rope_head_dim
    expected = 2 * tokens * m.q_lora_rank * heads_per_tp * q_head_dim
    assert op.roofline_spec().flops == expected   # 2*1*1536*16*192 = 9,437,184


def test_kv_a_proj_with_mqa_handcheck():
    """kv_a_proj_with_mqa: h → kv_lora_rank + qk_rope_head_dim, output to KV cache."""
    m, deployment, runtime = _deepseek_v3()
    tokens = 1
    ops = _layer_ops(m, deployment, runtime, tokens=tokens, phase="decode")
    op = _find_by_subtype(ops, "kv_a_proj_with_mqa")
    expected_oc = m.kv_lora_rank + m.rope_head_dim
    expected_flops = 2 * tokens * m.hidden_dim * expected_oc
    f = op.roofline_spec()
    assert f.flops == expected_flops   # 2*1*7168*576 = 8,257,536
    # 输出在 KV cache, 不在 activation
    assert f.store_kv_cache > 0
    assert f.store_act == 0


def test_kv_b_proj_handcheck():
    """kv_b_proj: kv_lora_rank → heads_per_tp × (qk_nope + v_head_dim), compute-time."""
    m, deployment, runtime = _deepseek_v3()
    tokens = 1
    ops = _layer_ops(m, deployment, runtime, tokens=tokens, phase="decode")
    op = _find_by_subtype(ops, "kv_b_proj")
    heads_per_tp = m.num_heads // deployment.parallelism.tp
    expected_oc = heads_per_tp * (m.qk_nope_head_dim + m.v_head_dim)
    expected_flops = 2 * tokens * m.kv_lora_rank * expected_oc
    f = op.roofline_spec()
    assert f.flops == expected_flops   # 2*1*512*16*(128+128) = 4,194,304
    assert f.store_act > 0
    assert f.store_kv_cache == 0


def test_o_proj_uses_v_head_dim_not_head_dim():
    """o_proj 的 input dim 应 = heads_per_tp × v_head_dim, 不是 heads_per_tp × head_dim."""
    m, deployment, runtime = _deepseek_v3()
    tokens = 1
    ops = _layer_ops(m, deployment, runtime, tokens=tokens, phase="decode")
    op = _find_by_subtype(ops, "o_proj")
    heads_per_tp = m.num_heads // deployment.parallelism.tp
    expected_ic = heads_per_tp * m.v_head_dim
    expected_flops = 2 * tokens * expected_ic * m.hidden_dim
    assert op.roofline_spec().flops == expected_flops    # 2*1*16*128*7168 = 29,360,128
    # 防御: 用错 head_dim=56, flops 会差 ~2.3×
    wrong_flops = 2 * tokens * heads_per_tp * m.head_dim * m.hidden_dim
    assert op.roofline_spec().flops != wrong_flops


# ---- q_lora_rank == 0 路径 ----

def test_mla_without_q_lora_uses_single_q_proj():
    """kv_lora_rank > 0 但 q_lora_rank == 0 时, Q 走单 q_proj."""
    m, deployment, runtime = _v3_lite_no_q_lora(tp=1)
    assert m.q_lora_rank == 0
    ops = _layer_ops(m, deployment, runtime, tokens=1, phase="decode")
    assert _has_subtype(ops, "q_proj")
    assert not _has_subtype(ops, "q_a_proj")
    assert not _has_subtype(ops, "q_b_proj")
    assert _has_subtype(ops, "kv_a_proj_with_mqa")
    assert _has_subtype(ops, "kv_b_proj")


# ---- 总 attention proj flops sanity (防回退到旧 head_dim bug) ----

def test_total_attn_proj_flops_significantly_higher_than_pre_fix():
    """总 attention proj flops (5 个 op) 应 ≥ 70M, 显著高于旧 bug ~42M."""
    m, deployment, runtime = _deepseek_v3()
    ops = _layer_ops(m, deployment, runtime, tokens=1, phase="decode")
    subtypes = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
    total = sum(_find_by_subtype(ops, s).roofline_spec().flops for s in subtypes)
    assert total > 70e6
    assert total < 80e6


# ---- MLA prefill FlashAttn KV IO (audit 修复) ----

def test_mla_prefill_new_token_does_not_read_kv_cache():
    """MLA 新 token prefill 不应读 KV cache (K/V 来自 kv_b_proj staging act, 不读 cache)."""
    m, deployment, runtime = _deepseek_v3(tp=8)
    ops = _layer_ops(m, deployment, runtime, tokens=128, phase="prefill")
    attn = next(op for op in ops if op.op_kind == "attention")
    assert attn.roofline_spec().load_kv_cache == 0


def test_mla_prefill_kv_staging_in_load_act_not_kv_cache():
    """MLA prefill 的 K/V staging (kv_b_proj 输出) 必须进 load_act, 用 a_byte."""
    m, deployment, runtime = _deepseek_v3(tp=8)
    ops = _layer_ops(m, deployment, runtime, tokens=128, phase="prefill")
    attn = next(op for op in ops if op.op_kind == "attention")
    heads_per_tp = m.num_heads // deployment.parallelism.tp
    qk_dim = m.qk_nope_head_dim + m.rope_head_dim
    v_dim = m.v_head_dim
    min_staging = int(128 * (qk_dim + v_dim) * 1 * heads_per_tp * A_BYTE)
    assert attn.roofline_spec().load_act >= min_staging


def test_mla_prefill_chunked_reads_c_kv_cache():
    """MLA chunked prefill 跨 step: prior context 应读 c_kv (single-head latent)."""
    m, deployment, runtime = _deepseek_v3(tp=8)
    ops = _chunked_prefill_layer_ops(m, deployment, runtime, current_tokens=128, total_ctx=1024)
    attn = next(op for op in ops if op.op_kind == "attention")
    # prior_ctx = 896, c_kv 维 = kv_latent_dim = 576
    expected_min = int(896 * 576 * 1 * 2.0 * 0.5)
    expected_max = int(896 * 576 * 1 * 2.0 * 2.0)
    assert expected_min <= attn.roofline_spec().load_kv_cache <= expected_max


def test_mla_prefill_kv_io_far_smaller_than_pre_fix():
    """新公式 KV cache IO 比旧 bug (num_heads × qk+v × kv_byte) 至少小 100×."""
    m, deployment, runtime = _deepseek_v3(tp=8)
    ops = _layer_ops(m, deployment, runtime, tokens=128, phase="prefill")
    attn = next(op for op in ops if op.op_kind == "attention")
    heads_per_tp = m.num_heads // deployment.parallelism.tp
    qk_dim = m.qk_nope_head_dim + m.rope_head_dim
    v_dim = m.v_head_dim
    old_bug_io = 128 * (qk_dim + v_dim) * heads_per_tp * 2.0
    assert attn.roofline_spec().load_kv_cache * 100 < old_bug_io


def test_mla_uses_dedicated_op_not_generic_fused_attention():
    """MLA 应该用 dedicated attention path: name='mla_attention' + tags 含 'mla'."""
    m, deployment, runtime = _deepseek_v3(tp=8)
    ops_dec = _layer_ops(m, deployment, runtime, tokens=1, phase="decode")
    attn_dec = next(op for op in ops_dec if op.op_kind == "attention")
    assert attn_dec.name == "mla_attention"
    assert "mla" in attn_dec.tags

    ops_pre = _layer_ops(m, deployment, runtime, tokens=128, phase="prefill")
    attn_pre = next(op for op in ops_pre if op.op_kind == "attention")
    assert attn_pre.name == "mla_attention"
    assert "mla" in attn_pre.tags


# ---- Non-MLA (Qwen GQA) prefill KV IO 不受影响 ----

def test_non_mla_prefill_kv_io_unchanged():
    """Qwen3 GQA prefill 的 KV cache IO 不应受 MLA 修复影响 (走 Qwen template, 不走 DeepSeek)."""
    from llm_infer_sim.core.cost.engine import build_qwen_dense_roofline_engine
    qwen = make_model_config(
        name="Qwen3-test",
        hidden_dim=4096, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=11008, num_layers=36, vocab_size=151936,
    )
    deployment = DeploymentProfile.flat(tp=1)
    runtime = RuntimeProfile.flat()
    hw = get_hardware_profile("RTX_4090")
    engine = build_qwen_dense_roofline_engine(qwen, deployment, runtime, hw)
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=128, context_len=0,
        )],
        num_prefill_tokens=128, total_scheduled_tokens=128,
        num_prefill_requests=1,
    )
    step = StepShape.from_workload(wl, "eager")
    plan = engine.model.forward(step)
    attn = next(op for op in plan.ops if op.op_kind == "attention" and op.layer_idx == 0)
    # GQA flash: kv_io = n_blocks_r × seqlen × head_dim × num_kv_heads × kv_byte × 2 > 0
    assert attn.roofline_spec().load_kv_cache > 0
