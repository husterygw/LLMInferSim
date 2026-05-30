"""CalibrationProvenance — 校准数据来源元信息 (config_plan §4.6)。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationProvenance:
    hardware: str = ""
    captured_at: str = ""
    source_version: str = ""
