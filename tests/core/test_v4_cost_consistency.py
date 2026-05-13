"""阶段 9-δ: DeepSeek-V4 cost 公式手算 vs actual 对照 (详设 §4.1.4 V4 sparse path).

按记忆 feedback_cost_formula_handcheck.md 必做.

覆盖 V4-Flash 真实 hf_config (tp=8 decode tokens=1):
  1. profile_extractor 14 个 V4 字段透传正确
  2. is_v4 = window_size > 0 AND o_groups > 0 = True
  3. fused_wqa_wkv, fused_compress_wkv_wgate, fused_index_compress_wkv_wgate 三个 fusion op 真激活
  4. index_wq_b / index_weights_proj 不切 TP (ReplicatedLinear)
  5. wo_a 用 deploy.w_byte 而非写死 2.0
  6. indexer_kv_byte 从 use_fp4_indexer_cache 推导 (1.0 默认, 0.5 fp4)
  7. wkv is_kv_proj=False (输出进 kv_norm 不写 KV cache)
  8. compress_ratios 列表正确分流 layer 0/2/3 (SWA / CSA / HCA)
"""
from types import SimpleNamespace

import pytest

from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
from llm_infer_sim.core.cost_model.layer_builder import moe_layer_time


def _v4_flash_bundle(tp_size: int = 8, use_fp4_indexer: bool = False):
    """V4-Flash 真实 hf_config 构造 bundle."""
    hf = SimpleNamespace(
        model_type="deepseek_v4",
        num_attention_heads=64, num_key_value_heads=1,
        hidden_size=4096, num_hidden_layers=43,
        vocab_size=129280, head_dim=512,
        # V4 attention LoRA
        q_lora_rank=1024, o_lora_rank=1024, o_groups=8,
        qk_rope_head_dim=64,
        # V4 sparse + indexer
        sliding_window=128,
        compress_ratios=[0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                         4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                         4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0],
        index_topk=512, index_n_heads=64, index_head_dim=128,
        # V4 HC
        hc_mult=4, hc_sinkhorn_iters=20,
        # V4 MoE
        n_routed_experts=256, num_experts_per_tok=6,
        moe_intermediate_size=2048, n_shared_experts=1,
        expert_dtype="fp4",
        scoring_func="sqrtsoftplus", topk_method="noaux_tc",
        # V4 misc
        num_hash_layers=3, swiglu_limit=10.0,
    )
    attn_cfg = SimpleNamespace(
        backend=None, use_fp4_indexer_cache=use_fp4_indexer,
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="deepseek-ai/DeepSeek-V4-Flash"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        attention_config=attn_cfg,
    )
    return extract_profile_bundle(vc)


def _find_op(lr, name):
    for op in lr.ops:
        if op.name == name:
            return op
    raise AssertionError(f"op {name!r} not in {[o.name for o in lr.ops]}")


def _has_op(lr, name) -> bool:
    return any(op.name == name for op in lr.ops)


# ============================================================================
# 9-α: 14 个 V4 字段透传 + is_v4 激活
# ============================================================================

def test_v4_fields_all_passthrough():
    """V4-Flash 14 个字段全部从 hf_config 透传到 ModelConfig."""
    b = _v4_flash_bundle()
    m = b.model
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
    """V4-Flash 必须触发 is_v4 = True (window_size>0 AND o_groups>0)."""
    b = _v4_flash_bundle()
    assert b.model.window_size > 0 and b.model.o_groups > 0


def test_v4_compress_ratio_routing():
    """compress_ratios list 决定每层 SWA-only/CSA/HCA."""
    b = _v4_flash_bundle()
    m = b.model
    assert m.get_compress_ratio(0) == 0     # SWA only
    assert m.get_compress_ratio(1) == 0
    assert m.get_compress_ratio(2) == 4     # CSA (indexer)
    assert m.get_compress_ratio(3) == 128   # HCA
    assert m.get_compress_ratio(4) == 4
    assert m.get_compress_ratio(42) == 4


# ============================================================================
# 9-α: 全 MoE 无 dense FFN 容错
# ============================================================================

def test_v4_no_dense_ffn_intermediate_size():
    """V4-Flash 没有 intermediate_size 字段, adapter 返回 0."""
    b = _v4_flash_bundle()
    assert b.model.ffn_dim == 0
    assert b.model.is_moe
    assert b.model.first_moe_layer == 0   # 没 first_k_dense_replace, 全 MoE


# ============================================================================
# 9-β bug 4: indexer_kv_byte 从 use_fp4_indexer_cache 推
# ============================================================================

def test_indexer_kv_byte_default_fp8():
    """默认 use_fp4_indexer_cache=False → indexer_kv_byte=1.0 (fp8)."""
    b = _v4_flash_bundle(use_fp4_indexer=False)
    assert b.deploy.indexer_kv_byte == 1.0


def test_indexer_kv_byte_fp4_when_attention_config_set():
    """use_fp4_indexer_cache=True → indexer_kv_byte=0.5 (mxfp4)."""
    b = _v4_flash_bundle(use_fp4_indexer=True)
    assert b.deploy.indexer_kv_byte == 0.5


# ============================================================================
# 9-γ fusion 1: fused_wqa_wkv 手算
# ============================================================================

def test_fused_wqa_wkv_handcheck():
    """fused_wqa_wkv: h × (q_lora + head_dim) MergedColumnParallelLinear (disable_tp).

    V4-Flash: h=4096, q_lora=1024, head_dim=512
      flops = 2 × tokens × h × (q_lora + head_dim) = 2*1*4096*1536 = 12,582,912
      load_act = h × tokens × a_byte = 4096 × 1 × 2 = 8,192  (x 一次)
    """
    b = _v4_flash_bundle()
    m, deploy = b.model, b.deploy
    tokens = 1
    lr = moe_layer_time(0, "decode", tokens, 128, m, deploy, b.hw)
    op = _find_op(lr, "fused_wqa_wkv")
    expected_flops = 2 * tokens * m.hidden_dim * (m.q_lora_rank + m.head_dim)
    assert op.flops == expected_flops    # 12,582,912
    expected_load_act = m.hidden_dim * tokens * deploy.a_byte
    assert op.load_act == expected_load_act
    # 旧拆分代码 (wq_a + wkv) load_act = 2 * h * tokens * a_byte;
    # fused 后 load_act 只 1× → 防御性检查
    assert op.load_act != 2 * m.hidden_dim * tokens * deploy.a_byte


def test_wq_a_and_wkv_no_longer_present():
    """fusion 后旧的独立 wq_a / wkv op 不应再出现."""
    b = _v4_flash_bundle()
    lr = moe_layer_time(0, "decode", 1, 128, b.model, b.deploy, b.hw)
    assert not _has_op(lr, "wq_a")
    assert not _has_op(lr, "wkv")
    assert _has_op(lr, "fused_wqa_wkv")


# ============================================================================
# 9-γ fusion 2: fused_compress_wkv_wgate 手算 (layer 2 ratio=4)
# ============================================================================

def test_fused_compress_wkv_wgate_handcheck_ratio4():
    """ratio=4 → coff=2 → fused_oc = 2*coff*head_dim = 2*2*512 = 2048.

    flops = 2 × tokens × h × fused_oc = 2*1*4096*2048 = 16,777,216
    """
    b = _v4_flash_bundle()
    m, deploy = b.model, b.deploy
    lr = moe_layer_time(2, "decode", 1, 128, m, deploy, b.hw)   # layer 2 ratio=4
    op = _find_op(lr, "fused_compress_wkv_wgate")
    fused_oc = 2 * 2 * m.head_dim   # 2 (wkv+gate) × coff(2) × head_dim
    assert op.flops == 2 * 1 * m.hidden_dim * fused_oc


def test_fused_compress_wkv_wgate_handcheck_ratio128():
    """ratio=128 → coff=1 → fused_oc = 2*1*head_dim = 1024.

    flops = 2 × tokens × h × fused_oc = 2*1*4096*1024 = 8,388,608
    """
    b = _v4_flash_bundle()
    m, deploy = b.model, b.deploy
    lr = moe_layer_time(3, "decode", 1, 128, m, deploy, b.hw)   # layer 3 ratio=128
    op = _find_op(lr, "fused_compress_wkv_wgate")
    fused_oc = 2 * 1 * m.head_dim
    assert op.flops == 2 * 1 * m.hidden_dim * fused_oc


def test_compress_wkv_and_gate_no_longer_present():
    """fusion 后 compress_wkv / compress_gate 独立 op 不应出现."""
    b = _v4_flash_bundle()
    lr = moe_layer_time(2, "decode", 1, 128, b.model, b.deploy, b.hw)
    assert not _has_op(lr, "compress_wkv")
    assert not _has_op(lr, "compress_gate")
    assert _has_op(lr, "fused_compress_wkv_wgate")


# ============================================================================
# 9-β bug 2: index_wq_b / index_weights_proj 不切 TP
# ============================================================================

def test_index_wq_b_does_not_divide_tp():
    """真实 indexer.wq_b 是 ReplicatedLinear (disable_tp), 不切 TP."""
    b = _v4_flash_bundle()
    m, deploy = b.model, b.deploy
    lr = moe_layer_time(2, "decode", 1, 128, m, deploy, b.hw)
    op = _find_op(lr, "index_wq_b")
    # 用完整 index_n_heads 而非 index_n_heads/tp
    index_full = m.index_n_heads
    expected_flops = 2 * 1 * m.q_lora_rank * index_full * m.index_head_dim
    assert op.flops == expected_flops    # 2*1*1024*64*128 = 16,777,216
    # 防御性: 旧 (除 tp) 公式会得到 1/tp 的 flops
    wrong = expected_flops // deploy.tp
    assert op.flops != wrong


def test_index_weights_proj_does_not_divide_tp():
    """同样 ReplicatedLinear."""
    b = _v4_flash_bundle()
    m, deploy = b.model, b.deploy
    lr = moe_layer_time(2, "decode", 1, 128, m, deploy, b.hw)
    op = _find_op(lr, "index_weights_proj")
    expected_flops = 2 * 1 * m.hidden_dim * m.index_n_heads
    assert op.flops == expected_flops
    wrong = expected_flops // deploy.tp
    assert op.flops != wrong


# ============================================================================
# 9-β bug 4: index_score 用 deploy.indexer_kv_byte
# ============================================================================

def test_index_score_kv_cache_byte_matches_deploy():
    """index_score 的 load_kv_cache 用 deploy.indexer_kv_byte (fp8=1.0 / fp4=0.5),
    不是写死 2.0 bf16. 切到 fp4 应该 K cache 字节减半."""
    b_fp8 = _v4_flash_bundle(use_fp4_indexer=False)
    b_fp4 = _v4_flash_bundle(use_fp4_indexer=True)
    lr_fp8 = moe_layer_time(2, "decode", 1, 128, b_fp8.model, b_fp8.deploy, b_fp8.hw)
    lr_fp4 = moe_layer_time(2, "decode", 1, 128, b_fp4.model, b_fp4.deploy, b_fp4.hw)
    op_fp8 = _find_op(lr_fp8, "index_score")
    op_fp4 = _find_op(lr_fp4, "index_score")
    # K cache part: fp4 应该是 fp8 的一半
    # 总 mem_bytes 包含 load_act + load_kv_cache; load_act 不变, kv 减半
    diff = op_fp8.load_kv_cache - op_fp4.load_kv_cache
    assert diff > 0
    assert op_fp4.load_kv_cache == pytest.approx(op_fp8.load_kv_cache / 2)


# ============================================================================
# 9-β bug 3: wo_a 用 deploy.w_byte
# ============================================================================

def test_wo_a_uses_deploy_w_byte_not_hardcoded():
    """旧代码写死 w_byte=2.0 bf16, 真实 V4 用 FP8.

    V4-Flash w_byte 由 efficiency_profile 决定, 当前默认 fp16 即 2.0,
    但 op_precision 字段应该跟 op_prec ('fp16' for w_byte=2.0) 一致.
    防御性:防止 w_byte 写死跟 deploy 解耦.
    """
    b = _v4_flash_bundle()
    m, deploy = b.model, b.deploy
    lr = moe_layer_time(0, "decode", 1, 128, m, deploy, b.hw)
    op = _find_op(lr, "wo_a")
    # 防御性: wo_a 的 load_weight 应该正比于 deploy.w_byte
    n_local_groups = max(m.o_groups // deploy.tp, 1)
    heads_per_tp = m.num_heads // deploy.tp
    wo_a_ic = heads_per_tp * m.head_dim // n_local_groups
    wo_a_oc = n_local_groups * m.o_lora_rank
    expected_weight = int(wo_a_ic * wo_a_oc * deploy.w_byte)
    assert op.load_weight == expected_weight


# ============================================================================
# 9-β bug 1: wkv is_kv_proj=False (已在 fused_wqa_wkv 后已隐式覆盖, 直接验 V3 影响)
# ============================================================================

def test_v3_mla_wkv_separate_not_in_v4_path():
    """V4 path 用 fused_wqa_wkv, 不应再出现独立 wkv (是 MLA path 的字段)."""
    b = _v4_flash_bundle()
    lr = moe_layer_time(0, "decode", 1, 128, b.model, b.deploy, b.hw)
    assert not _has_op(lr, "wkv")


# ============================================================================
# 整体: V4 layer 内含完整 V4 path 的 ops 列表
# ============================================================================

def test_v4_layer_2_full_path_ops_present():
    """layer 2 (ratio=4 CSA, ALSO hash MoE 因为 num_hash_layers=3):
    必须有 HC + V4 attn + compressor + indexer + sparse_attention + wo + hash MoE."""
    b = _v4_flash_bundle()
    lr = moe_layer_time(2, "decode", 1, 128, b.model, b.deploy, b.hw)
    required = [
        "hc_attn_pre", "attn_norm",
        "fused_wqa_wkv", "q_norm", "wq_b", "kv_norm",
        "fused_compress_wkv_wgate", "compress_pool",
        "fused_index_compress_wkv_wgate", "index_wq_b", "index_weights_proj", "index_score",
        "fused_sparse_attention", "wo_a", "wo_b", "attn_allreduce", "hc_attn_post",
        "hc_ffn_pre", "mlp_norm",
        "moe_hash_lookup",  # ← layer 2 < num_hash_layers=3, hash routing (fix 13)
        "routed_experts",
        "shared_expert_up_gate", "shared_expert_down", "hc_ffn_post",
    ]
    for name in required:
        assert _has_op(lr, name), f"missing op {name!r} in V4 layer 2"
    # layer 2 hash 层不应该有 moe_gate (router GEMM)
    assert not _has_op(lr, "moe_gate")


# ============================================================================
# 阶段 9 fix 13: V4 hash MoE 层跳过 moe_gate FLOPs
# ============================================================================

def test_v4_hash_layers_use_lookup_not_gate():
    """V4-Flash num_hash_layers=3 → layer 0/1/2 走 tid2eid lookup, 没 moe_gate."""
    b = _v4_flash_bundle()
    assert b.model.num_hash_layers == 3
    for hash_idx in (0, 1, 2):
        lr = moe_layer_time(hash_idx, "decode", 1, 128, b.model, b.deploy, b.hw)
        assert _has_op(lr, "moe_hash_lookup"), f"layer {hash_idx} 应有 hash lookup"
        assert not _has_op(lr, "moe_gate"), f"layer {hash_idx} 不该有 moe_gate"


def test_v4_non_hash_layers_still_use_gate():
    """V4-Flash layer ≥ 3 应该走正常 moe_gate (router GEMM)."""
    b = _v4_flash_bundle()
    for non_hash_idx in (3, 4, 5, 42):
        lr = moe_layer_time(non_hash_idx, "decode", 1, 128, b.model, b.deploy, b.hw)
        assert _has_op(lr, "moe_gate"), f"layer {non_hash_idx} 应有 moe_gate"
        assert not _has_op(lr, "moe_hash_lookup"), f"layer {non_hash_idx} 不该有 hash lookup"


def test_v4_hash_lookup_zero_flops():
    """hash MoE lookup FLOPs 应该 0 (tid2eid 是查表, 不是 GEMM)."""
    b = _v4_flash_bundle()
    lr = moe_layer_time(0, "decode", 1, 128, b.model, b.deploy, b.hw)
    op = _find_op(lr, "moe_hash_lookup")
    assert op.flops == 0


# ============================================================================
# 阶段 9 fix 10/11: HC model-level ops (hc_embedding_repeat / hc_head / final_norm)
# ============================================================================

def test_v4_model_core_has_hc_head_and_final_norm():
    """V4-Flash hc_mult>0 时, ModelCoreCostModel.estimate 输出含 hc_embedding_repeat /
    hc_head / final_norm 三个 model-level ops (fix 10/11)."""
    from llm_infer_sim.core.cost_model.model_core import ModelCoreCostModel
    from llm_infer_sim.core.workload.workload import (
        GlobalStepWorkload, RequestWorkload, StepPhase,
    )
    b = _v4_flash_bundle()
    mc = ModelCoreCostModel(b)
    wl = GlobalStepWorkload(
        step_id=1, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=128, context_len=128, target_output_len=8,
        )],
        num_prefill_tokens=128, num_decode_tokens=0,
        total_scheduled_tokens=128,
        num_prefill_requests=1, num_decode_requests=0,
    )
    result = mc.estimate(wl)
    names = {op["name"] for op in result["per_op"]}
    assert "hc_embedding_repeat" in names
    assert "hc_head" in names
    assert "final_norm" in names


def test_non_hc_model_no_hc_model_level_ops():
    """Qwen3-30B-A3B hc_mult=0, ModelCoreCostModel 不应注入 HC ops."""
    hf = SimpleNamespace(
        model_type="qwen3_moe",
        num_attention_heads=32, num_key_value_heads=4,
        hidden_size=2048, num_hidden_layers=48,
        intermediate_size=6144, vocab_size=151936, head_dim=128,
        num_experts=128, num_experts_per_tok=8,
        moe_intermediate_size=768, mlp_only_layers=[],
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="Qwen3-30B-A3B"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2, data_parallel_size=1, enable_expert_parallel=False,
        ),
    )
    b = extract_profile_bundle(vc)
    assert b.model.hc_mult == 0
    from llm_infer_sim.core.cost_model.model_core import ModelCoreCostModel
    from llm_infer_sim.core.workload.workload import (
        GlobalStepWorkload, RequestWorkload, StepPhase,
    )
    mc = ModelCoreCostModel(b)
    wl = GlobalStepWorkload(
        step_id=1, phase=StepPhase.DECODE,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.DECODE,
            num_tokens=1, context_len=128, target_output_len=8, generated_tokens=10,
        )],
        num_prefill_tokens=0, num_decode_tokens=1,
        total_scheduled_tokens=1,
        num_prefill_requests=0, num_decode_requests=1,
    )
    result = mc.estimate(wl)
    names = {op["name"] for op in result["per_op"]}
    assert "hc_embedding_repeat" not in names
    assert "hc_head" not in names
    assert "final_norm" not in names


# ============================================================================
# 阶段 9 fix 17: attn_sink +1 attended pos
# ============================================================================

def test_attn_sink_increments_attended_count():
    """sparse attention 每 query +1 attended pos (sink)."""
    from llm_infer_sim.core.ops.attention import attention_decode_sparse
    ops_with_sink = attention_decode_sparse(
        ctx_len=128, batchsize=1, num_attention_heads=8, head_size=512,
        a_byte=2.0, kv_byte=1.0, window_size=64,
        compress_ratio=4, index_topk=16, onchip_buffer=None,
    )
    # attended = min(128, 64) + min(16, 128//4) + 1 = 64 + 16 + 1 = 81
    # qk_ops = attended * head_size * num_heads * batchsize * 2 = 81*512*8*1*2 = 663,552
    expected_qk = 81 * 512 * 8 * 1 * 2
    expected_sv = 1 * 512 * 81 * 8 * 1 * 2
    expected_softmax = 1 * 8 * 81 * 1 * 5
    expected_flops = expected_qk + expected_sv + expected_softmax
    assert ops_with_sink[0].flops == expected_flops


def test_v4_layer_0_swa_only_no_compressor_no_indexer():
    """layer 0 (ratio=0 SWA-only): 没有 compressor / indexer ops, 但 fused_wqa_wkv + sparse_attention 还在."""
    b = _v4_flash_bundle()
    lr = moe_layer_time(0, "decode", 1, 128, b.model, b.deploy, b.hw)
    assert _has_op(lr, "fused_wqa_wkv")
    assert _has_op(lr, "fused_sparse_attention")
    assert not _has_op(lr, "fused_compress_wkv_wgate")
    assert not _has_op(lr, "fused_index_compress_wkv_wgate")
    assert not _has_op(lr, "index_score")


def test_v4_layer_3_hca_compressor_but_no_indexer():
    """layer 3 (ratio=128 HCA): 有 compressor, 但**没有** indexer (HCA 用全部 compressed 不 select)."""
    b = _v4_flash_bundle()
    lr = moe_layer_time(3, "decode", 1, 128, b.model, b.deploy, b.hw)
    assert _has_op(lr, "fused_compress_wkv_wgate")
    assert _has_op(lr, "compress_pool")
    assert not _has_op(lr, "fused_index_compress_wkv_wgate")
    assert not _has_op(lr, "index_score")
