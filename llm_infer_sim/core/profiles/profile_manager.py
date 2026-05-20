"""ProfileBundle —— 框架无关的三件套打包 (详设 §4.8.3 数据类)。

阶段 3.5 重构后:
  - ProfileBundle 数据类留在 core (框架无关)
  - 从 vLLM 配置抽取的实现搬到 adapters/vllm/profile_extractor.py
    (详设 §1.1 架构分层: core 完全框架无关)

详设引用:
  §1.1   架构分层 (core 不感知 vllm)
  §4.8.3 ProfileBundle 数据类
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_infer_sim.core.profiles.backend_profile import (
    BackendExecutionProfile,
    default_backend_profile,
)
from llm_infer_sim.core.profiles.deploy import LegacyDeployConfig
from llm_infer_sim.core.profiles.efficiency_profile import EfficiencyProfile
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_adapters import UnsupportedModelError
from llm_infer_sim.core.profiles.model_config import ModelConfig


@dataclass
class ProfileBundle:
    """框架无关的三件套打包供 cost model 直接使用。

    构造入口:
      - adapters/vllm/profile_extractor.extract_profile_bundle(vllm_config)
      - (将来) adapters/sglang/profile_extractor.extract_profile_bundle(sgl_config)
      - 直接 ProfileBundle(...) 用于 standalone / 测试
    """

    model: ModelConfig
    deploy: LegacyDeployConfig
    hw: HardwareConfig
    efficiency: EfficiencyProfile
    backend: BackendExecutionProfile = field(default_factory=default_backend_profile)


__all__ = ["ProfileBundle", "UnsupportedModelError"]
