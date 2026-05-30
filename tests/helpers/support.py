"""共享测试 helper: legacy config 构造的唯一入口 (config_plan Step H)。

集中 legacy ModelConfig / HardwareConfig 构造, 给 Step I 冻结/重命名 legacy
dataclass 时留一个改动点。helper 是 thin pass-through: 透传 kwargs, 产出与各
测试原构造逐字段一致 (byte-identical)。

注意: config 必须是本模块第一个 llm_infer_sim import —— 预解析 models 包的
import cycle (graph → step_plan → operators → context → models.config → ...)。
"""
from __future__ import annotations

from llm_infer_sim.core.models.config import ModelConfig
from llm_infer_sim.core.hardware.device import HardwareConfig


def make_model_config(**overrides) -> ModelConfig:
    """构造 legacy ModelConfig (透传 kwargs)。"""
    return ModelConfig(**overrides)


def make_hardware_config(**overrides) -> HardwareConfig:
    """构造 legacy HardwareConfig (透传 kwargs)。"""
    return HardwareConfig(**overrides)
