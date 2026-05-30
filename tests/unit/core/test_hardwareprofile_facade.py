"""config_plan Step E: HardwareProfile 扁平 read facade 与 HardwareConfig duck-type 等价。

锁住: RooflineBackend / RooflineAnalyzer / communication.py 用 HardwareProfile 直读, 结果
与等价 flat HardwareConfig byte-identical。facade 的 effective_* / 拓扑缩放逐式复刻
HardwareConfig 的 __post_init__ 派生 + effective_* 公式。
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.roofline import communication as comm
from llm_infer_sim.core.hardware.device import HardwareConfig, HardwareProfile
from llm_infer_sim.core.hardware.registry import KNOWN_PROFILES, get_hardware_config

_SCALAR_PROPS = (
    "effective_peak_flops", "effective_peak_int8", "effective_peak_bf16",
    "effective_peak_fp8", "effective_peak_fp4", "effective_vector_flops",
    "effective_mem_bandwidth", "effective_comm_bandwidth", "effective_inter_bw",
    "ridge_point", "has_fp4_tc",
    "intra_node_size", "intra_node_protocol_efficiency", "comm_step_latency",
    "inter_node_latency", "has_nvlink_sharp", "enable_nvls_model",
)

_NAMES = sorted(KNOWN_PROFILES.keys())
_CASES = pytest.mark.parametrize("name", _NAMES)


@_CASES
def test_scalar_facade_matches_legacy(name):
    cfg: HardwareConfig = get_hardware_config(name)
    prof = HardwareProfile.from_legacy(cfg)
    for p in _SCALAR_PROPS:
        assert getattr(prof, p) == getattr(cfg, p), p


@_CASES
def test_effective_intra_bw_matches_legacy(name):
    cfg = get_hardware_config(name)
    prof = HardwareProfile.from_legacy(cfg)
    for n in (1, 2, 4, 8, 16, 32):
        for hint in ("concentrated", "balanced"):
            assert prof.effective_intra_bw(n, hint) == (
                cfg.effective_intra_bw(n, hint)
            ), (name, n, hint)


@_CASES
def test_allreduce_time_matches_legacy(name):
    cfg = get_hardware_config(name)
    prof = HardwareProfile.from_legacy(cfg)
    for n in (2, 4, 8):
        for data in (4096.0, 1 << 20, 1 << 24):
            assert comm.allreduce_time(data, n, prof) == (
                comm.allreduce_time(data, n, cfg)
            ), (name, n, data)


@_CASES
def test_alltoall_time_matches_legacy(name):
    cfg = get_hardware_config(name)
    prof = HardwareProfile.from_legacy(cfg)
    for n in (2, 4, 8):
        for data in (1 << 16, 1 << 22):
            assert comm.alltoall_time(data, n, prof) == (
                comm.alltoall_time(data, n, cfg)
            ), (name, n, data)
