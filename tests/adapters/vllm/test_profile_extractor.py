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
