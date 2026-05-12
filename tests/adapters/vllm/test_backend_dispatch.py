"""阶段 3.5: vLLM Backend → mixed_mode 映射 (详设 §4.8.1.1)。

实现搬到 adapters/vllm/profile_extractor.py 后, 本测试相应搬到 tests/adapters/vllm/。

覆盖:
  1. 主流 backend (FLASH_ATTN / FLASHINFER) → unified_ragged
  2. backend=None → flash_attn_auto / unified_ragged (vLLM platform 自动选)
  3. MLA 系列 → unified_ragged 占位 (阶段 8 真实验证)
  4. 非 NVIDIA backend (ROCM_*) → NotImplementedError
  5. 未知 enum (假冒 name) → NotImplementedError

不依赖完整 vLLM 启动, 用 SimpleNamespace 模拟 vllm_config.attention_config.backend。
"""
from types import SimpleNamespace

import pytest

from llm_infer_sim.adapters.vllm.profile_extractor import (
    _VLLM_BACKEND_MODE_MAP,
    _VLLM_UNSUPPORTED_BACKENDS,
    _extract_backend_profile,
)
from llm_infer_sim.core.profiles.backend_profile import BackendExecutionProfile


def _mock_vllm_config(backend_name: str | None):
    """构造一个最小 vllm_config, 只含 attention_config.backend。

    backend_name=None: 模拟 vllm_config.attention_config.backend = None。
    backend_name=str: 模拟一个具名 enum (SimpleNamespace 带 .name 属性即够)。
    """
    backend = SimpleNamespace(name=backend_name) if backend_name else None
    return SimpleNamespace(attention_config=SimpleNamespace(backend=backend))


def test_flash_attn_maps_to_unified_ragged():
    cfg = _mock_vllm_config("FLASH_ATTN")
    profile = _extract_backend_profile(cfg)
    assert isinstance(profile, BackendExecutionProfile)
    assert profile.name == "flash_attn"
    assert profile.mixed_attention.mode == "unified_ragged"


def test_flashinfer_maps_to_unified_ragged():
    cfg = _mock_vllm_config("FLASHINFER")
    profile = _extract_backend_profile(cfg)
    assert profile.name == "flashinfer"
    assert profile.mixed_attention.mode == "unified_ragged"


def test_backend_none_uses_platform_default():
    """backend=None: vLLM platform 自动选, 阶段 3.5 简化为 unified_ragged。"""
    cfg = _mock_vllm_config(None)
    profile = _extract_backend_profile(cfg)
    assert profile.name == "flash_attn_auto"
    assert profile.mixed_attention.mode == "unified_ragged"


def test_mla_backends_are_placeholders():
    """MLA 系列阶段 8 才真实验证, 阶段 3.5 先映射 unified_ragged 占位。"""
    for name in ("FLASH_ATTN_MLA", "FLASHMLA", "FLASHINFER_MLA", "TRITON_MLA"):
        profile = _extract_backend_profile(_mock_vllm_config(name))
        assert profile.mixed_attention.mode == "unified_ragged", name
        assert profile.name == name.lower()


def test_rocm_backend_rejected():
    """非 NVIDIA backend 应 fail-fast。"""
    cfg = _mock_vllm_config("ROCM_ATTN")
    with pytest.raises(NotImplementedError, match="ROCM_ATTN"):
        _extract_backend_profile(cfg)


def test_unknown_backend_rejected():
    """未在 _VLLM_BACKEND_MODE_MAP 中的新 enum 应 fail-fast 并提示加映射。"""
    cfg = _mock_vllm_config("HYPOTHETICAL_NEW_BACKEND_V42")
    with pytest.raises(NotImplementedError, match="Unknown attention backend"):
        _extract_backend_profile(cfg)


# ------- 映射表本身的一致性检查 -------

def test_backend_map_no_overlap_with_unsupported():
    """同一 backend 不能既在 supported map 又在 unsupported set。"""
    overlap = set(_VLLM_BACKEND_MODE_MAP.keys()) & _VLLM_UNSUPPORTED_BACKENDS
    assert not overlap, f"backend 名重叠: {overlap}"


def test_backend_map_modes_all_implemented():
    """_VLLM_BACKEND_MODE_MAP 的 mode 必须是 MixedAttentionEstimator 已实现的策略。"""
    implemented = {"split_kernels", "unified_ragged"}
    for backend_name, (_, mode) in _VLLM_BACKEND_MODE_MAP.items():
        assert mode in implemented, (
            f"backend {backend_name} 映射到 {mode}, 但 MixedAttentionEstimator "
            f"未实现该 mode。"
        )


def test_default_backend_profile_still_works():
    """default_backend_profile() 保留供 standalone 模式, 不应受改动影响。"""
    from llm_infer_sim.core.profiles.backend_profile import default_backend_profile
    profile = default_backend_profile()
    # 仍保持阶段 3 的默认 split_kernels (供 standalone fallback)
    assert profile.mixed_attention.mode == "split_kernels"
