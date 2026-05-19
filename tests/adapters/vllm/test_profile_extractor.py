"""extract_profile_bundle 解析正确性 (阶段 3.5 重构后从 ProfileManager 改名)。

不依赖 vLLM 真实 LLM 实例 —— 用 mock VllmConfig (只需要 model_config /
parallel_config / attention_config 三个子对象)。
"""
from types import SimpleNamespace

from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
from llm_infer_sim.core.profiles.profile_manager import ProfileBundle


def _make_vllm_config(hf_config, model_id="dummy"):
    """合成最小 VllmConfig 形态: 只暴露 extract_profile_bundle 用到的子字段。"""
    model_config = SimpleNamespace(
        hf_config=hf_config,
        model=model_id,
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        data_parallel_size=1,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
    )


def test_qwen3_4b_parse():
    """阶段 2 退出条件: 能正确解析 Qwen3 hf_config."""
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=2560,
        num_hidden_layers=36,
        intermediate_size=9728,
        vocab_size=151936,
        head_dim=128,                # Qwen3 显式 head_dim ≠ hidden / num_heads
    )
    vllm_cfg = _make_vllm_config(hf, model_id="Qwen/Qwen3-4B-Instruct-2507")
    bundle = extract_profile_bundle(vllm_cfg)

    assert isinstance(bundle, ProfileBundle)
    m = bundle.model
    assert m.num_layers == 36
    assert m.hidden_dim == 2560
    assert m.num_heads == 32
    assert m.num_kv_heads == 8       # GQA
    assert m.head_dim == 128         # 来自显式 head_dim 字段, 而非 hidden / num_heads = 80
    assert m.ffn_dim == 9728
    assert m.vocab_size == 151936
    assert not m.is_moe
    assert m.kv_lora_rank == 0       # 非 MLA


def test_opt125m_parse():
    """opt-125m: MHA dense, head_dim 来自 hidden / num_heads = 64."""
    hf = SimpleNamespace(
        model_type="opt",
        num_attention_heads=12,
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,                # opt 用 ffn_dim 字段
        vocab_size=50272,
    )
    vllm_cfg = _make_vllm_config(hf, model_id="facebook/opt-125m")
    bundle = extract_profile_bundle(vllm_cfg)

    m = bundle.model
    assert m.num_layers == 12
    assert m.hidden_dim == 768
    assert m.num_heads == 12
    assert m.num_kv_heads == 12      # MHA = num_kv == num_heads
    assert m.head_dim == 768 // 12   # 没显式 head_dim, fallback 到 hidden/num_heads
    assert m.ffn_dim == 3072
    assert m.vocab_size == 50272


def test_efficiency_placeholder_all_ones():
    """阶段 1/2: EfficiencyProfile placeholder 全 1.0 (无 calibration).

    阶段 X.1 Plan B 后字段重命名 compute_efficiency → default_compute;
    placeholder() 行为不变.
    """
    hf = SimpleNamespace(
        model_type="opt",
        num_attention_heads=12, hidden_size=768, num_hidden_layers=12,
        ffn_dim=3072, vocab_size=50272,
    )
    bundle = extract_profile_bundle(_make_vllm_config(hf))
    assert bundle.efficiency.default_compute == 1.0
    assert bundle.efficiency.default_mem == 1.0
    assert bundle.efficiency.default_comm == 1.0
    assert bundle.efficiency.w_byte == 2.0    # fp16
    assert bundle.efficiency.entries == {}    # placeholder 不带 lookup entry


def test_unsupported_model_raises():
    import pytest

    from llm_infer_sim.core.profiles.model_adapters import UnsupportedModelError

    hf = SimpleNamespace(model_type="nonexistent_family")
    with pytest.raises(UnsupportedModelError):
        extract_profile_bundle(_make_vllm_config(hf))


def test_qwen3_30b_a3b_moe_parse():
    """阶段 5: Qwen3-30B-A3B MoE 字段被正确读 (兼容 Qwen num_experts 命名)。"""
    hf = SimpleNamespace(
        model_type="qwen3_moe",
        num_attention_heads=32,
        num_key_value_heads=4,
        hidden_size=2048,
        num_hidden_layers=48,
        intermediate_size=6144,        # dense FFN dim, MoE layer 不用
        vocab_size=151936,
        head_dim=128,
        # MoE fields (Qwen 命名)
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
        mlp_only_layers=[],            # 全部 MoE 层
    )
    bundle = extract_profile_bundle(_make_vllm_config(hf, "Qwen/Qwen3-30B-A3B"))
    m = bundle.model
    assert m.is_moe                    # ✓ Qwen num_experts 字段被识别
    assert m.num_experts == 128
    assert m.num_activated_experts == 8
    assert m.expert_dim == 768
    assert m.num_shared_experts == 0   # A3B 没有 shared
    assert m.first_moe_layer == 0      # mlp_only_layers=[] → 第 0 层就是 MoE
    # 验证 is_moe_layer(i) 在所有层都为 True
    assert all(m.is_moe_layer(i) for i in range(m.num_layers))


def test_deepseek_naming_still_works():
    """DeepSeek 命名 (n_routed_experts / n_shared_experts / first_k_dense_replace) 不能破坏。"""
    hf = SimpleNamespace(
        model_type="qwen3_moe",        # 用 qwen adapter, 但字段名走 DeepSeek 路径
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_size=2048,
        num_hidden_layers=24,
        intermediate_size=8192,
        vocab_size=102400,
        head_dim=128,
        n_routed_experts=64,
        num_experts_per_tok=6,
        moe_intermediate_size=1024,
        n_shared_experts=2,
        first_k_dense_replace=3,
    )
    bundle = extract_profile_bundle(_make_vllm_config(hf, "fake-deepseek-like"))
    m = bundle.model
    assert m.is_moe
    assert m.num_experts == 64
    assert m.num_shared_experts == 2
    assert m.first_moe_layer == 3      # 前 3 层是 dense


def test_qwen_shared_expert_intermediate_size_derived():
    """Qwen 旧 MoE (Qwen2-MoE) 用 shared_expert_intermediate_size, n_shared 推算得到。"""
    hf = SimpleNamespace(
        model_type="qwen2_moe",
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_size=2048,
        num_hidden_layers=24,
        intermediate_size=8192,
        vocab_size=102400,
        head_dim=128,
        num_experts=60,
        num_experts_per_tok=4,
        moe_intermediate_size=1408,
        shared_expert_intermediate_size=2816,   # = 2 × moe_intermediate_size
    )
    bundle = extract_profile_bundle(_make_vllm_config(hf, "fake-qwen2-moe"))
    m = bundle.model
    assert m.is_moe
    assert m.num_experts == 60
    assert m.num_shared_experts == 2   # 2816 / 1408 = 2


def test_deepseek_v3_parse():
    """阶段 8-α: DeepSeek-V3 hf_config (MLA + MoE + shared experts + Q-side LoRA)。"""
    hf = SimpleNamespace(
        model_type="deepseek_v3",
        num_attention_heads=128, num_key_value_heads=128,
        hidden_size=7168, num_hidden_layers=61,
        intermediate_size=18432, vocab_size=129280,
        # MLA
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=1536,
        # MoE
        n_routed_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        first_k_dense_replace=3,
    )
    bundle = extract_profile_bundle(_make_vllm_config(hf, "deepseek-ai/DeepSeek-V3"))
    m = bundle.model
    # 基础
    assert m.hidden_dim == 7168
    assert m.num_heads == 128
    assert m.num_layers == 61
    assert m.ffn_dim == 18432
    # MLA 字段
    assert m.kv_lora_rank == 512
    assert m.qk_nope_head_dim == 128
    assert m.rope_head_dim == 64
    assert m.kv_latent_dim == 512 + 64        # = kv_lora_rank + qk_rope_head_dim
    assert m.q_lora_rank == 1536              # ★ V3 Q-side LoRA, 阶段 8-α 新增透传
    # MoE
    assert m.is_moe
    assert m.num_experts == 256
    assert m.num_activated_experts == 8
    assert m.expert_dim == 2048
    assert m.num_shared_experts == 1          # ★ V3 shared experts
    assert m.first_moe_layer == 3             # ★ 前 3 层 dense FFN
    # 层路由
    assert not m.is_moe_layer(0)
    assert not m.is_moe_layer(2)
    assert m.is_moe_layer(3)
    assert m.is_moe_layer(60)


def test_extracted_backend_profile_for_no_attention_config():
    """attention_config 缺失时 fallback 到 flash_attn_auto / unified_ragged。"""
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=2560, num_hidden_layers=36,
        intermediate_size=9728, vocab_size=151936, head_dim=128,
    )
    bundle = extract_profile_bundle(_make_vllm_config(hf))
    assert bundle.backend.name == "flash_attn_auto"
    assert bundle.backend.mixed_attention.mode == "unified_ragged"


# ---------------------------------------------------------------------------
# §10.5 8.5 FP8 / Quantization 解析 (quant_method + cache_dtype + activation_scheme)
# ---------------------------------------------------------------------------

def _make_vllm_config_with(quant_cfg=None, cache_dtype=None):
    """带 quantization_config / cache_config 的 mock VllmConfig."""
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=2560, num_hidden_layers=36,
        intermediate_size=9728, vocab_size=151936, head_dim=128,
        quantization_config=quant_cfg,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="dummy"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype) if cache_dtype else None,
    )


def test_quant_method_fp8_switches_w_byte_and_a_byte():
    """quant_method 含 "fp8" + activation_scheme="dynamic" → w_byte=a_byte=1.0."""
    qcfg = {"quant_method": "fp8", "activation_scheme": "dynamic"}
    bundle = extract_profile_bundle(_make_vllm_config_with(quant_cfg=qcfg))
    assert bundle.deploy.w_byte == 1.0
    assert bundle.deploy.a_byte == 1.0


def test_quant_method_deepseek_v4_fp8_substring_match():
    """vLLM 把 quant_method 改写成 "deepseek_v4_fp8", 子串匹配应仍切 w_byte=1.0."""
    qcfg = {"quant_method": "deepseek_v4_fp8", "activation_scheme": "dynamic"}
    bundle = extract_profile_bundle(_make_vllm_config_with(quant_cfg=qcfg))
    assert bundle.deploy.w_byte == 1.0
    assert bundle.deploy.a_byte == 1.0


def test_quant_method_fp4_overrides_w_a_byte():
    """fp4 优先匹配 (避免 "fp4" 被 "fp8" 撞上, 子串顺序很关键)."""
    qcfg = {"quant_method": "nvfp4", "activation_scheme": "dynamic"}
    bundle = extract_profile_bundle(_make_vllm_config_with(quant_cfg=qcfg))
    assert bundle.deploy.w_byte == 0.5
    assert bundle.deploy.a_byte == 0.5


def test_no_quant_config_defaults_to_fp16():
    """无 quantization_config → 默认 fp16 (w_byte=a_byte=2.0)."""
    bundle = extract_profile_bundle(_make_vllm_config_with())
    assert bundle.deploy.w_byte == 2.0
    assert bundle.deploy.a_byte == 2.0


def test_activation_scheme_missing_keeps_a_byte_default():
    """quant_method=fp8 但无 activation_scheme → 只切 w_byte, a_byte 保留默认 (保守)."""
    qcfg = {"quant_method": "fp8"}
    bundle = extract_profile_bundle(_make_vllm_config_with(quant_cfg=qcfg))
    assert bundle.deploy.w_byte == 1.0
    assert bundle.deploy.a_byte == 2.0


def test_cache_dtype_fp8_switches_kv_byte():
    """cache_config.cache_dtype="fp8" → kv_byte=1.0."""
    bundle = extract_profile_bundle(_make_vllm_config_with(cache_dtype="fp8"))
    assert bundle.deploy.kv_byte == 1.0


def test_cache_dtype_fp8_e4m3_substring_match():
    """cache_dtype="fp8_e4m3" 子串匹配 → kv_byte=1.0."""
    bundle = extract_profile_bundle(_make_vllm_config_with(cache_dtype="fp8_e4m3"))
    assert bundle.deploy.kv_byte == 1.0


def test_cache_dtype_auto_keeps_default():
    """cache_dtype="auto" → kv_byte 保持默认 fp16 (跟随 model dtype)."""
    bundle = extract_profile_bundle(_make_vllm_config_with(cache_dtype="auto"))
    assert bundle.deploy.kv_byte == 2.0


def test_cache_dtype_int8_switches_kv_byte():
    bundle = extract_profile_bundle(_make_vllm_config_with(cache_dtype="int8"))
    assert bundle.deploy.kv_byte == 1.0


# ---------------------------------------------------------------------------
# base_w_byte / base_a_byte (non-quantized 层 dtype from model_config.dtype)
# ---------------------------------------------------------------------------

def _make_vllm_config_with_dtype(model_dtype, quant_cfg=None, cache_dtype=None):
    """支持 model_config.dtype 的 helper."""
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=2560, num_hidden_layers=36,
        intermediate_size=9728, vocab_size=151936, head_dim=128,
        quantization_config=quant_cfg,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="dummy", dtype=model_dtype),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype) if cache_dtype else None,
    )


def test_base_dtype_bfloat16_gives_2_byte():
    """model_config.dtype = torch.bfloat16 → base_w/a_byte = 2.0."""
    import torch
    bundle = extract_profile_bundle(_make_vllm_config_with_dtype(torch.bfloat16))
    assert bundle.deploy.base_w_byte == 2.0
    assert bundle.deploy.base_a_byte == 2.0


def test_base_dtype_float16_gives_2_byte():
    import torch
    bundle = extract_profile_bundle(_make_vllm_config_with_dtype(torch.float16))
    assert bundle.deploy.base_w_byte == 2.0


def test_base_dtype_float32_gives_4_byte():
    import torch
    bundle = extract_profile_bundle(_make_vllm_config_with_dtype(torch.float32))
    assert bundle.deploy.base_w_byte == 4.0


def test_base_dtype_decouples_from_quant_method():
    """关键: fp8 量化模型 lm_head/embed/norm 仍走 base = model dtype.

    DeepSeek-V3 部署形态: dtype=bfloat16 + quant_method=fp8 →
      w_byte=1.0 (主体 fp8)
      base_w_byte=2.0 (lm_head/embed/norm 仍 bf16)
      base_a_byte=2.0 (activation peak buffer 仍 bf16)
    """
    import torch
    qcfg = {"quant_method": "fp8", "activation_scheme": "dynamic"}
    vc = _make_vllm_config_with_dtype(torch.bfloat16, quant_cfg=qcfg)
    bundle = extract_profile_bundle(vc)
    assert bundle.deploy.w_byte == 1.0           # 主体 fp8
    assert bundle.deploy.a_byte == 1.0           # GEMM input fp8
    assert bundle.deploy.base_w_byte == 2.0      # lm_head/embed/norm bf16
    assert bundle.deploy.base_a_byte == 2.0      # peak buffer bf16


def test_base_dtype_none_fallback_to_2():
    """model_config.dtype 缺失 (老 mock / 没 dtype) → fallback 2.0."""
    bundle = extract_profile_bundle(_make_vllm_config_with())
    assert bundle.deploy.base_w_byte == 2.0
    assert bundle.deploy.base_a_byte == 2.0


def test_torch_dtype_to_byte_helper():
    """_torch_dtype_to_byte 直接覆盖各种 dtype 字符串."""
    import torch
    from llm_infer_sim.adapters.vllm.profile_extractor import _torch_dtype_to_byte
    assert _torch_dtype_to_byte(torch.bfloat16) == 2.0
    assert _torch_dtype_to_byte(torch.float16) == 2.0
    assert _torch_dtype_to_byte(torch.float32) == 4.0
    assert _torch_dtype_to_byte(torch.float64) == 4.0
    assert _torch_dtype_to_byte(None) == 2.0     # default
    assert _torch_dtype_to_byte(None, default=4.0) == 4.0


def test_torch_dtype_to_byte_fp8_string_match():
    """torch.float8_e4m3fn 等 fp8 dtype → 1.0."""
    from llm_infer_sim.adapters.vllm.profile_extractor import _torch_dtype_to_byte
    # 用字符串 (不所有 torch 版本都有 float8 类型)
    assert _torch_dtype_to_byte("torch.float8_e4m3fn") == 1.0
    assert _torch_dtype_to_byte("torch.int8") == 1.0


# ---------------------------------------------------------------------------
# Non-quantized modules 解析 (compressed-tensors.ignore + awq.modules_to_not_convert)
# ---------------------------------------------------------------------------

def test_classify_no_ignore_list():
    """无 ignore/modules_to_not_convert → 空列表."""
    from llm_infer_sim.adapters.vllm.profile_extractor import _classify_non_quantized_modules
    qcfg = {"quant_method": "fp8"}
    cov, unh = _classify_non_quantized_modules(qcfg)
    assert cov == [] and unh == []


def test_classify_compressed_tensors_ignore_lm_head():
    """compressed-tensors ignore 含 "re:.*lm_head" → covered (已在 base 路径)."""
    from llm_infer_sim.adapters.vllm.profile_extractor import _classify_non_quantized_modules
    qcfg = {
        "quant_method": "compressed-tensors",
        "ignore": ["re:.*lm_head", "model.embed_tokens"],
    }
    cov, unh = _classify_non_quantized_modules(qcfg)
    assert len(cov) == 2
    assert unh == []


def test_classify_awq_modules_to_not_convert():
    """awq 用 modules_to_not_convert 字段 (不是 ignore)."""
    from llm_infer_sim.adapters.vllm.profile_extractor import _classify_non_quantized_modules
    qcfg = {"quant_method": "awq", "modules_to_not_convert": ["lm_head"]}
    cov, unh = _classify_non_quantized_modules(qcfg)
    assert cov == ["lm_head"]
    assert unh == []


def test_classify_unhandled_specific_linear():
    """ignore 含特定 Linear (例某层 q_proj), 不在 base 集合 → unhandled."""
    from llm_infer_sim.adapters.vllm.profile_extractor import _classify_non_quantized_modules
    qcfg = {
        "quant_method": "compressed-tensors",
        "ignore": ["re:.*lm_head", "model.layers.0.self_attn.q_proj"],
    }
    cov, unh = _classify_non_quantized_modules(qcfg)
    assert cov == ["re:.*lm_head"]
    assert unh == ["model.layers.0.self_attn.q_proj"]


def test_classify_norm_patterns():
    """各种 norm 名称都应被识别为 covered."""
    from llm_infer_sim.adapters.vllm.profile_extractor import _classify_non_quantized_modules
    qcfg = {
        "quant_method": "fp8",
        "ignore": ["input_layernorm", "post_attention_layernorm", "rms_norm",
                   "final_layernorm"],
    }
    cov, unh = _classify_non_quantized_modules(qcfg)
    assert len(cov) == 4
    assert unh == []


def test_extractor_propagates_classification_to_deploy():
    """profile_extractor 把分类结果填到 DeployConfig."""
    import torch
    qcfg = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "ignore": ["re:.*lm_head"],
    }
    vc = _make_vllm_config_with_dtype(torch.bfloat16, quant_cfg=qcfg)
    bundle = extract_profile_bundle(vc)
    assert "re:.*lm_head" in bundle.deploy.covered_non_quantized
    assert bundle.deploy.unhandled_non_quantized_modules == []


def test_extractor_warns_on_unhandled():
    """unhandled pattern 触发 warning + 写入 deploy."""
    import warnings
    import torch
    qcfg = {
        "quant_method": "compressed-tensors",
        "ignore": ["model.layers.5.mlp.gate_up_proj"],  # 不在 base 集合
    }
    vc = _make_vllm_config_with_dtype(torch.bfloat16, quant_cfg=qcfg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bundle = extract_profile_bundle(vc)
        # 至少 1 个 warning 含 unhandled pattern
        msgs = [str(item.message) for item in w]
        assert any("model.layers.5.mlp.gate_up_proj" in m for m in msgs)
    assert "model.layers.5.mlp.gate_up_proj" in bundle.deploy.unhandled_non_quantized_modules
