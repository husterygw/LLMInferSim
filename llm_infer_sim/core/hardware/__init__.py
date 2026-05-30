"""core/hardware — 硬件域配置 (config_plan §3/§4.4)。"""
from llm_infer_sim.core.hardware.communication import CommunicationProfile
from llm_infer_sim.core.hardware.device import (
    ComputeSpec,
    HardwareConfig,
    HardwareProfile,
    MemorySpec,
)
from llm_infer_sim.core.hardware.registry import (
    get_hardware_config,
    get_hardware_profile,
)
from llm_infer_sim.core.hardware.topology import NodeTopology

__all__ = [
    "ComputeSpec",
    "MemorySpec",
    "HardwareConfig",
    "HardwareProfile",
    "NodeTopology",
    "CommunicationProfile",
    "get_hardware_config",
    "get_hardware_profile",
]
