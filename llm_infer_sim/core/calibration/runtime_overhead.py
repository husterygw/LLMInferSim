"""RuntimeOverheadCalibration — 框架/kernel 常数开销校准 (config_plan §4.6 + Step G)。

kernel_overhead (per-op eager dispatch 开销, us → s) 现归此 (Step G 从 HardwareProfile
搬出); cost backend 经 calibration.runtime_overhead.kernel_overhead 读。prefill worker
overhead 仍由 RuntimeProfile.execution 携带。默认 {} = 无 per-op 开销。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeOverheadCalibration:
    kernel_overhead: dict = field(default_factory=dict)
