"""ProfileBundle —— 框架无关的三件套打包 (V3 §4.8.3).

ProfileBundle.deploy = V3 §4.6 DeployConfig (含可选 pd).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_infer_sim.core.profiles.backend_profile import (
    BackendExecutionProfile,
    default_backend_profile,
)
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.efficiency_profile import EfficiencyProfile
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_adapters import UnsupportedModelError
from llm_infer_sim.core.profiles.model_config import ModelConfig


@dataclass
class ProfileBundle:
    """框架无关的三件套打包供 StepCostEngine 直接使用.

    构造入口:
      - adapters/vllm/profile_extractor.extract_profile_bundle(vllm_config)
      - 直接 ProfileBundle(...) 用于 standalone / 测试
    """

    model: ModelConfig
    deploy: DeployConfig
    hw: HardwareConfig
    efficiency: EfficiencyProfile
    backend: BackendExecutionProfile = field(default_factory=default_backend_profile)


__all__ = ["ProfileBundle", "UnsupportedModelError"]
