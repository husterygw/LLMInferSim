"""阶段 3 C 块: VirtualPlatform Feature Gate 6 类 fail-fast (详设 §7.5.2)。

构造 SimpleNamespace 模拟 vllm_config 子结构, 仅测 _check_unsupported_features
逻辑 (避免依赖完整 vllm 启动)。
"""
from types import SimpleNamespace

import pytest

from llm_infer_sim.adapters.vllm.virtual_platform import VirtualPlatform


def _bare_config(**overrides):
    """构造一个 baseline 干净的 vllm_config: 所有不支持 feature 都不开。"""
    cfg = SimpleNamespace(
        lora_config=None,
        speculative_config=None,
        decoding_config=SimpleNamespace(guided_decoding_backend=None),
        kv_transfer_config=None,
        attention_config=SimpleNamespace(backend=None),  # 阶段 3.5: 第 7 类
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            max_logprobs=0,
        ),
    )
    for k, v in overrides.items():
        if k.startswith("model_"):
            setattr(cfg.model_config, k[len("model_"):], v)
        elif k.startswith("decoding_"):
            setattr(cfg.decoding_config, k[len("decoding_"):], v)
        elif k.startswith("attention_"):
            setattr(cfg.attention_config, k[len("attention_"):], v)
        else:
            setattr(cfg, k, v)
    return cfg


def test_clean_config_passes():
    cfg = _bare_config()
    # 不应 raise
    VirtualPlatform._check_unsupported_features(cfg)


def test_lora_rejected():
    cfg = _bare_config(lora_config=SimpleNamespace(max_lora_rank=8))
    with pytest.raises(ValueError, match="LoRA"):
        VirtualPlatform._check_unsupported_features(cfg)


def test_speculative_decoding_rejected():
    cfg = _bare_config(speculative_config=SimpleNamespace(method="ngram"))
    with pytest.raises(ValueError, match="Speculative"):
        VirtualPlatform._check_unsupported_features(cfg)


def test_multimodal_rejected():
    cfg = _bare_config(model_is_multimodal_model=True)
    with pytest.raises(ValueError, match="Multi-modal"):
        VirtualPlatform._check_unsupported_features(cfg)


def test_guided_decoding_rejected():
    cfg = _bare_config(decoding_guided_decoding_backend="outlines")
    with pytest.raises(ValueError, match="Guided decoding"):
        VirtualPlatform._check_unsupported_features(cfg)


def test_logprobs_rejected():
    cfg = _bare_config(model_max_logprobs=5)
    with pytest.raises(ValueError, match="Logprobs"):
        VirtualPlatform._check_unsupported_features(cfg)


def test_kv_transfer_rejected_unless_env_set(monkeypatch):
    cfg = _bare_config(kv_transfer_config=SimpleNamespace(kind="lookup"))
    monkeypatch.delenv("VLLM_INFER_SIM_ALLOW_PD_DISAGG", raising=False)
    with pytest.raises(ValueError, match="KV transfer"):
        VirtualPlatform._check_unsupported_features(cfg)

    # 环境变量显式设 1 后允许
    monkeypatch.setenv("VLLM_INFER_SIM_ALLOW_PD_DISAGG", "1")
    VirtualPlatform._check_unsupported_features(cfg)


def test_unsupported_attention_backend_rejected():
    """阶段 3.5 第 7 类: 非 NVIDIA backend platform 启动期就拦。"""
    cfg = _bare_config(attention_backend=SimpleNamespace(name="ROCM_ATTN"))
    with pytest.raises(ValueError, match="ROCM_ATTN"):
        VirtualPlatform._check_unsupported_features(cfg)


def test_unknown_attention_backend_rejected():
    cfg = _bare_config(
        attention_backend=SimpleNamespace(name="HYPOTHETICAL_NEW_BACKEND")
    )
    with pytest.raises(ValueError, match="Unknown attention backend"):
        VirtualPlatform._check_unsupported_features(cfg)


def test_flash_attn_backend_passes():
    """主流 backend 不应被 feature gate 拦下。"""
    cfg = _bare_config(attention_backend=SimpleNamespace(name="FLASH_ATTN"))
    VirtualPlatform._check_unsupported_features(cfg)


def test_multiple_violations_reported_together():
    cfg = _bare_config(
        lora_config=SimpleNamespace(),
        model_max_logprobs=3,
    )
    with pytest.raises(ValueError) as exc_info:
        VirtualPlatform._check_unsupported_features(cfg)
    msg = str(exc_info.value)
    assert "LoRA" in msg
    assert "Logprobs" in msg
