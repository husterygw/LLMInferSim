"""校准注册表 (config_plan Step G)。

按硬件名给出对应的 CalibrationProfile —— 校准数据独立于硬件物理 spec, 但取值
依赖具体硬件 (kernel_overhead / moe_efficiency 是 per-hw 标定结果)。

历史上这些值挂在 HardwareConfig 上 (calibrated=True 时填), Step G 搬到这里:
hardware registry 只管物理 spec + efficiency 系数, calibration registry 管标定 knob。
"""
from __future__ import annotations

from llm_infer_sim.core.calibration.moe_efficiency import rtx_4090_moe_efficiency_v1
from llm_infer_sim.core.calibration.profile import CalibrationProfile
from llm_infer_sim.core.hardware.registry import PROFILE_ALIASES


def get_calibration_profile(name: str, calibrated: bool = True) -> CalibrationProfile:
    """按硬件名返回 CalibrationProfile。

    calibrated=False 或无标定数据的硬件 → 纯占位 (moe_efficiency=None, kernel_overhead={}).
    当前只有 RTX_4090 有 MoE 标定 (Phase 5.B neutral, 仅 trace metadata 不改 latency)。
    """
    canonical = PROFILE_ALIASES.get(name, name)
    if calibrated and canonical == "RTX_4090":
        return CalibrationProfile(moe_efficiency=rtx_4090_moe_efficiency_v1())
    return CalibrationProfile()
