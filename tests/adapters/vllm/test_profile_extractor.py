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
    """阶段 1/2: EfficiencyProfile 全 1.0 (无 calibration)."""
    hf = SimpleNamespace(
        model_type="opt",
        num_attention_heads=12, hidden_size=768, num_hidden_layers=12,
        ffn_dim=3072, vocab_size=50272,
    )
    bundle = extract_profile_bundle(_make_vllm_config(hf))
    assert bundle.efficiency.compute_efficiency == 1.0
    assert bundle.efficiency.mem_efficiency == 1.0
    assert bundle.efficiency.comm_efficiency == 1.0
    assert bundle.efficiency.w_byte == 2.0    # fp16


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
