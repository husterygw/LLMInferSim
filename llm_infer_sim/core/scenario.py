"""SimulationScenario — 唯一完整配置输入 (config_plan §2)。

由明确的领域对象组成 (model / deployment / hardware / runtime / calibration)。
生产入口 adapters/vllm/profile_extractor.extract_scenario 直接装配。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_infer_sim.core.calibration.profile import CalibrationProfile
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.hardware.device import HardwareProfile
from llm_infer_sim.core.models.config import ModelProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile


@dataclass(frozen=True)
class SimulationScenario:
    model: ModelProfile
    deployment: DeploymentProfile
    hardware: HardwareProfile
    runtime: RuntimeProfile
    calibration: CalibrationProfile = field(default_factory=CalibrationProfile)
