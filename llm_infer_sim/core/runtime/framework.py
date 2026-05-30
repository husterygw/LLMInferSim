"""FrameworkProfile — 推理框架标识 (config_plan §4.5)。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameworkProfile:
    name: str = "vllm"
    version: str | None = None
