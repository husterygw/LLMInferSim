"""RooflineCalibration — roofline 公式的校准系数 (config_plan §4.6)。

历史上 EfficiencyProfile 承载 per-op efficiency 查表 (roofline_predicted/real),
但该数据线违反校准方法论已 retire 并删除。此处保留占位 (全 1.0 = 纯 roofline),
等未来 MeasuredOperatorDB 落地后填真测数据。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RooflineCalibration:
    default_compute: float = 1.0
    default_mem: float = 1.0
    default_comm: float = 1.0
