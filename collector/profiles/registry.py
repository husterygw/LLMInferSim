"""profile_name → ProfileSpec lookup. 给 CLI 用."""
from __future__ import annotations

import importlib

from collector.profiles._dims import ProfileSpec


# 注册新 profile: 这里加一行 (profile module 必须导出 PROFILE 常量)
_KNOWN_PROFILES: dict[str, str] = {
    "qwen3_4b":      "collector.profiles.qwen3_4b",
    "qwen3_30b_a3b": "collector.profiles.qwen3_30b_a3b",
}


def list_profile_names() -> list[str]:
    return sorted(_KNOWN_PROFILES.keys())


def load_profile(name: str) -> ProfileSpec:
    """按 profile_name 加载 ProfileSpec. 未知 name raise KeyError."""
    if name not in _KNOWN_PROFILES:
        raise KeyError(
            f"Unknown profile {name!r}. Known: {list_profile_names()}"
        )
    mod = importlib.import_module(_KNOWN_PROFILES[name])
    profile = getattr(mod, "PROFILE", None)
    if not isinstance(profile, ProfileSpec):
        raise TypeError(
            f"Profile module {_KNOWN_PROFILES[name]!r} must export PROFILE: ProfileSpec"
        )
    return profile
