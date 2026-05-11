"""Model family adapter dispatcher + 各 family getter 输出。

不依赖 vLLM, 只构造一个有最小字段的 dummy hf_config 对象。
"""
import pytest

from llm_infer_sim.core.profiles.model_adapters import (
    ADAPTERS,
    UnsupportedModelError,
    get_adapter,
)


class _OPTConfig:
    # opt-125m 关键字段 (来自 facebook/opt-125m 的 config.json)
    num_attention_heads = 12
    hidden_size = 768
    num_hidden_layers = 12
    ffn_dim = 3072
    vocab_size = 50272


class _Qwen3Config:
    # Qwen3-4B-Instruct-2507 关键字段
    num_attention_heads = 32
    num_key_value_heads = 8
    hidden_size = 2560
    num_hidden_layers = 36
    intermediate_size = 9728
    vocab_size = 151936
    head_dim = 128


def test_dispatcher_known_keys():
    for key in ("opt", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe"):
        assert key in ADAPTERS, f"missing dispatcher entry: {key}"


def test_dispatcher_unknown_raises():
    with pytest.raises(UnsupportedModelError):
        get_adapter("not_a_real_family")


def test_opt_adapter_getters():
    a = get_adapter("opt")
    cfg = _OPTConfig()
    assert a.get_num_attention_heads(cfg) == 12
    assert a.get_hidden_size(cfg) == 768
    # OPT 是 MHA, num_key_value_heads getter 直接返回 num_attention_heads
    assert a.get_num_key_value_heads(cfg) == 12
    assert a.get_num_hidden_layers(cfg) == 12
    assert a.get_intermediate_size(cfg) == 3072  # opt 的 FFN 字段叫 ffn_dim
    assert a.get_vocab_size(cfg) == 50272


def test_qwen_adapter_getters_gqa():
    a = get_adapter("qwen3")
    cfg = _Qwen3Config()
    assert a.get_num_attention_heads(cfg) == 32
    assert a.get_num_key_value_heads(cfg) == 8  # GQA: kv != heads
    assert a.get_hidden_size(cfg) == 2560
    assert a.get_num_hidden_layers(cfg) == 36
    assert a.get_intermediate_size(cfg) == 9728
    assert a.get_vocab_size(cfg) == 151936


def test_qwen_adapter_falls_back_to_attention_heads_when_no_kv():
    """Qwen 适配器: 没 num_key_value_heads 字段时退化到 num_attention_heads (MHA 行为)。"""
    class _NoKV:
        num_attention_heads = 16
        hidden_size = 1024
        num_hidden_layers = 8
        intermediate_size = 4096
        vocab_size = 32000

    a = get_adapter("qwen3")
    assert a.get_num_key_value_heads(_NoKV()) == 16
