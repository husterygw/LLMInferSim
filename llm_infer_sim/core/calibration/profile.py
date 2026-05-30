"""CalibrationProfile — 校准域聚合 (config_plan §4.6 + Step G)。

校准参数独立于硬件物理规格 (Step G: kernel_overhead / moe_efficiency 从 HardwareConfig
搬到这里, 不再 masquerade 成硬件物理 spec)。kernel_overhead 在 runtime_overhead,
moe_efficiency 是 MoE 校准 knob (topk/dispatch overhead + grouped_gemm_efficiency)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from llm_infer_sim.core.calibration.moe_efficiency import MoEEfficiencyProfile
from llm_infer_sim.core.calibration.provenance import CalibrationProvenance
from llm_infer_sim.core.calibration.roofline import RooflineCalibration
from llm_infer_sim.core.calibration.runtime_overhead import RuntimeOverheadCalibration


@dataclass(frozen=True)
class CalibrationProfile:
    roofline: RooflineCalibration = field(default_factory=RooflineCalibration)
    runtime_overhead: RuntimeOverheadCalibration = field(
        default_factory=RuntimeOverheadCalibration
    )
    # MoE calibration knob (moe_plan §5.A). None = 无 calibration (退回纯 roofline)。
    moe_efficiency: Optional[MoEEfficiencyProfile] = None
    provenance: CalibrationProvenance = field(default_factory=CalibrationProvenance)
