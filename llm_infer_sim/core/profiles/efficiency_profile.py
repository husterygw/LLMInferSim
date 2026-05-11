"""EfficiencyProfile — 硬件效率系数表 (placeholder for stage 1).

阶段 1 : 全 1.0, 无 calibration。这意味着 cost model 输出 = 纯 roofline 上界。
        系统方案 §6 顶部约定: 阶段 0~9 全是 placeholder, 阶段 X 才填 calibration。

阶段 X (真实平台校准) 起 : 该 schema 升级为
    硬件 × 模型 × dtype × kernel-version 四维 lookup table
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EfficiencyProfile:
    """阶段 1 占位: 所有 efficiency 都是 1.0, cost = pure roofline 上界。"""

    # 三大类 efficiency (与 llm-viewer HardwareConfig 同字段名以便后续 patch)
    compute_efficiency: float = 1.0
    mem_efficiency: float = 1.0
    comm_efficiency: float = 1.0

    # bytes per element (跟随系统量化设置, 阶段 1 默认 fp16)
    w_byte: float = 2.0
    a_byte: float = 2.0
    kv_byte: float = 2.0

    @property
    def w_bit(self) -> int:
        return int(self.w_byte * 8)

    @property
    def a_bit(self) -> int:
        return int(self.a_byte * 8)

    @property
    def kv_bit(self) -> int:
        return int(self.kv_byte * 8)

    @classmethod
    def placeholder(cls) -> "EfficiencyProfile":
        """阶段 1 默认: 全 1.0 + fp16."""
        return cls()

    def apply_to(self, hw) -> None:
        """把 efficiency 系数应用到 llm-viewer HardwareConfig 实例上。

        llm-viewer HardwareConfig 已有 compute_efficiency / mem_efficiency /
        comm_efficiency 字段, 默认 1.0。阶段 X 起会用 calibrated 值覆盖。
        """
        hw.compute_efficiency = self.compute_efficiency
        hw.mem_efficiency = self.mem_efficiency
        hw.comm_efficiency = self.comm_efficiency
