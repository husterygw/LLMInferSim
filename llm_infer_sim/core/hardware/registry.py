"""Hardware profile registry (config_plan §4.4)。

承载 preset 表 (KNOWN_PROFILES / PROFILE_ALIASES) + 两个 accessor:
  - get_hardware_config(name) → 扁平 HardwareConfig (canonical 查表逻辑, cost 路径直读)
  - get_hardware_profile(name) → 结构化 HardwareProfile (scenario 域入口, 包 from_legacy)

Values are converted from origin/k8s-deploy:hardwares/hardware_params.py into the
local HardwareConfig format. Numeric expressions intentionally stay close to the
source hardware_params.py values; only field names and interconnect bandwidth units
are adapted for HardwareConfig.
"""
from __future__ import annotations

from llm_infer_sim.core.hardware.device import HardwareConfig, HardwareProfile
from llm_infer_sim.core.hardware.communication import rtx_4090_nccl_communication


# vector_flops is the non-Tensor-Core/vector peak used for element-wise
# norm/activation/softmax roofline. For NVIDIA profiles this uses published
# FP32 CUDA-core peak when available; export/SKU variants inherit the matching
# base silicon value.


KNOWN_PROFILES: dict[str, dict] = {
    "A100": dict(
        peak_flops_fp16=312e12,
        peak_flops_bf16=312e12,
        peak_flops_int8=624e12,
        peak_flops_fp8=0.0,
        peak_flops_fp4=0.0,
        vector_flops=19.5e12,
        has_fp4_tc=False,
        mem_bandwidth=1555e9,
        mem_capacity_gb=40,
        onchip_buffer=27648e3,
        intra_node_bandwidth=600e9,
        intra_node_protocol_efficiency=0.67,  # NVLink 3 busbw ~200/300 GB/s, https://github.com/NVIDIA/nccl-tests/issues/149
        comm_step_latency=0.0,
        inter_node_bandwidth=2.5e+10,
        inter_node_latency=0.0,
    ),
    "H100": dict(
        peak_flops_fp16=1979e12 / 2,
        peak_flops_bf16=1979e12 / 2,
        peak_flops_int8=3958e12 / 2,
        peak_flops_fp8=3958e12 / 2,
        peak_flops_fp4=0.0,
        vector_flops=67e12,
        has_fp4_tc=False,
        mem_bandwidth=3350e9,
        mem_capacity_gb=80,
        onchip_buffer=33792e3,
        intra_node_bandwidth=900e9,
        intra_node_protocol_efficiency=0.68,  # NVLink 4 busbw ~300/450 GB/s, https://github.com/NVIDIA/nccl-tests/issues/212 + https://ai-hpc.org/en/guide/03-network/nccl-test
        has_nvlink_sharp=True,                # NVSwitch/NVLink SHARP support
        comm_step_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H800": dict(
        peak_flops_fp16=1979e12 / 2,
        peak_flops_bf16=1979e12 / 2,
        peak_flops_int8=3958e12 / 2,
        peak_flops_fp8=3958e12 / 2,
        peak_flops_fp4=0.0,
        vector_flops=67e12,
        has_fp4_tc=False,
        mem_bandwidth=3350e9,
        mem_capacity_gb=80,
        onchip_buffer=33792e3,
        intra_node_bandwidth=400e9,
        intra_node_protocol_efficiency=0.68,  # NVLink 4 busbw ~300/450 GB/s, https://github.com/NVIDIA/nccl-tests/issues/212 + https://ai-hpc.org/en/guide/03-network/nccl-test
        has_nvlink_sharp=True,                # NVSwitch/NVLink SHARP support
        comm_step_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H200": dict(
        peak_flops_fp16=1979e12 / 2,
        peak_flops_bf16=1979e12 / 2,
        peak_flops_int8=1979e12,
        peak_flops_fp8=3958e12 / 2,
        peak_flops_fp4=0.0,
        vector_flops=67e12,
        has_fp4_tc=False,
        mem_bandwidth=4800e9,
        mem_capacity_gb=141,
        onchip_buffer=33792e3,
        intra_node_bandwidth=900e9,
        intra_node_protocol_efficiency=0.68,  # NVLink 4 busbw ~300/450 GB/s, https://github.com/NVIDIA/nccl-tests/issues/212 + https://ai-hpc.org/en/guide/03-network/nccl-test
        has_nvlink_sharp=True,                # NVSwitch/NVLink SHARP support
        comm_step_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H20_96G": dict(
        peak_flops_fp16=148e12,
        peak_flops_bf16=148e12,
        peak_flops_int8=296e12,
        peak_flops_fp8=296e12,
        peak_flops_fp4=0.0,
        vector_flops=44e12,
        has_fp4_tc=False,
        mem_bandwidth=4.0e12,
        mem_capacity_gb=96,
        onchip_buffer=19968e3,
        intra_node_bandwidth=900e9,
        intra_node_protocol_efficiency=0.68,  # NVLink 4 busbw ~300/450 GB/s, https://github.com/NVIDIA/nccl-tests/issues/212 + https://ai-hpc.org/en/guide/03-network/nccl-test
        has_nvlink_sharp=True,                # NVSwitch/NVLink SHARP support
        comm_step_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "B200": dict(
        peak_flops_fp16=2250e12,
        peak_flops_bf16=2250e12,
        peak_flops_int8=4500e12,
        peak_flops_fp8=4500e12,
        peak_flops_fp4=9000e12,
        vector_flops=80e12,
        has_fp4_tc=True,
        mem_bandwidth=8.0e12,
        mem_capacity_gb=192,
        onchip_buffer=65536e3,
        intra_node_bandwidth=1800e9,
        has_nvlink_sharp=True,                # NVSwitch/NVLink SHARP support
        comm_step_latency=0.0,
        inter_node_bandwidth=1e+11,
        inter_node_latency=0.0,
    ),
    "B300": dict(
        peak_flops_fp16=2250e12,
        peak_flops_bf16=2250e12,
        peak_flops_int8=4500e12,
        peak_flops_fp8=5000e12,
        peak_flops_fp4=15000e12,
        vector_flops=6e15 / 72,
        has_fp4_tc=True,
        mem_bandwidth=8.0e12,
        mem_capacity_gb=288,
        onchip_buffer=65536e3,
        intra_node_bandwidth=1800e9,
        has_nvlink_sharp=True,                # NVSwitch/NVLink SHARP support
        comm_step_latency=0.0,
        inter_node_bandwidth=1e+11,
        inter_node_latency=0.0,
    ),
    "NGU800P": dict(
        peak_flops_fp16=1000e12,
        peak_flops_bf16=1000e12,
        peak_flops_int8=2000e12,
        peak_flops_fp8=2000e12,
        peak_flops_fp4=4000e12,
        vector_flops=33e12,
        has_fp4_tc=True,
        mem_bandwidth=2.0e12,
        mem_capacity_gb=192,
        onchip_buffer=20480e3,
        intra_node_bandwidth=1100e9,
        comm_step_latency=0.0,
        inter_node_bandwidth=0.0,
        inter_node_latency=0.0,
    ),
    "NGU800D": dict(
        peak_flops_fp16=1000e12,
        peak_flops_bf16=1000e12,
        peak_flops_int8=4000e12,
        peak_flops_fp8=4000e12,
        peak_flops_fp4=12000e12,
        vector_flops=33e12,   # 暂无准确数字
        has_fp4_tc=True,
        mem_bandwidth=10.0e12,
        mem_capacity_gb=288,
        onchip_buffer=20480e3,
        intra_node_bandwidth=1800e9,
        comm_step_latency=0.0,
        inter_node_bandwidth=0.0,
        inter_node_latency=0.0,
    ),
    "Ascend_950PR": dict(
        peak_flops_fp16=500e12,
        peak_flops_bf16=500e12,
        peak_flops_int8=1000e12,
        peak_flops_fp8=1000e12,
        peak_flops_fp4=2000e12,
        vector_flops=130e12,  # 暂时查不到，采用它对标的rubin nv144
        has_fp4_tc=True,
        mem_bandwidth=1.4e12,
        mem_capacity_gb=112,
        onchip_buffer=20480e3,
        intra_node_bandwidth=2e12,  # 2 TB/s, 参考 https://www.huawei.com/cn/news/2025/9/hc-xu-keynote-speech
        comm_step_latency=0.0,
        inter_node_bandwidth=0.0,
        inter_node_latency=0.0,
    ),
    "Ascend_950DT": dict(
        peak_flops_fp16=500e12,
        peak_flops_bf16=500e12,
        peak_flops_int8=1000e12,
        peak_flops_fp8=1000e12,
        peak_flops_fp4=2000e12,
        vector_flops=130e12,  # 暂时查不到，采用它对标的rubin nv144
        has_fp4_tc=True,
        mem_bandwidth=4.0e12,
        mem_capacity_gb=144,
        onchip_buffer=20480e3,
        intra_node_bandwidth=2e12,  # 2 TB/s, 参考 https://www.huawei.com/cn/news/2025/9/hc-xu-keynote-speech
        comm_step_latency=0.0,
        inter_node_bandwidth=0.0,
        inter_node_latency=0.0,
    ),
    # ----- Consumer cards -----
    # RTX 4090 (Ada Lovelace, SM89). Calibration target (详 §9.4.2 Plan B).
    # FP8 (E4M3) 在 Ada 上有原生 TC 支持; FP4 无 (推到 Blackwell).
    # 无 NVLink, 卡间走 PCIe 4.0 ×16 (~32 GB/s 双向).
    # 数据源: NVIDIA Ada whitepaper + 4090 product page.
    "RTX_4090": dict(
        peak_flops_fp16=165.2e12,         # BF16/FP16 TC dense
        peak_flops_bf16=165.2e12,
        peak_flops_int8=660.6e12,         # INT8 TC dense
        peak_flops_fp8=660.6e12,          # FP8 (E4M3) TC dense, 2× BF16
        peak_flops_fp4=0.0,               # Ada 不支持 FP4
        vector_flops=82.6e12,             # FP32 CUDA core
        has_fp4_tc=False,
        mem_bandwidth=1008e9,             # GDDR6X 21 Gbps × 384-bit ≈ 1008 GB/s
        mem_capacity_gb=24,
        onchip_buffer=32768e3,            # 128 SM × 256 KB register file (AD102)
        # PCIe 4.0 ×16: 32 GB/s 单向 / 64 GB/s 双向 nominal. 实测 AllReduce
        # effective β = nominal/2 × protocol_eff / n_per_root.
        # Phase 5: 拓扑感知, 跨 NUMA 部署自动有更高 effective β.
        # 实测验证 (详 scripts/measure_collectives.py + docs/COMMUNICATION_MODELING.md):
        #   n=2 cross_numa (n_per_root=1): β = 22.4 GB/s (max)
        #   n=2 same_numa  (n_per_root=2): β = 11.2 GB/s
        #   n=4 cross_numa (n_per_root=2): β = 11.2 GB/s
        #   n=4 same_numa  (n_per_root=4): β = 5.6 GB/s  ← Phase 5 校准点
        #   n=8 (必跨 NUMA, n_per_root=4): β = 5.6 GB/s
        intra_node_bandwidth=64e9,                # PCIe 4.0 ×16 双向 nominal
        intra_node_topology="pcie_shared_root",
        intra_node_gpus_per_root=4,               # 当前测试机 nvidia-smi topo -m 实测
        intra_node_num_roots=2,                   # dual NUMA, GPU 0-3 / 4-7
        gpu_to_root={                             # 显式映射(GPU id → NUMA id)
            0: 0, 1: 0, 2: 0, 3: 0,
            4: 1, 5: 1, 6: 1, 7: 1,
        },
        # protocol_efficiency 0.625: 独立实测 scripts/measure_allreduce.py
        # (TP=4 ring AllReduce β fit), 反推 0.625, 比公开 NVLink 0.67-0.68 略低
        # (PCIe 实际比 NVLink 多一层 root contention). 2026-05-15 测得.
        intra_node_protocol_efficiency=0.625,
        # comm_step_latency: 2026-05-17 scripts/measure_p2p_latency.py 实测 9.6 µs.
        # 方法: GPU 0 → 1 cudagraph round-trip 19.3 µs / 2 hops = 9.6 µs.
        # 这是 NCCL P2P kernel 内部 latency (含 SerDes + protocol sync), 不是裸 PCIe.
        # raw 数据: configs/calibration/raw/RTX_4090/2026-05-17/measure_p2p_latency.jsonl
        comm_step_latency=9.6e-6,
        has_nvlink_sharp=False,                    # Ada 没 NVLink/NVLS
        inter_node_bandwidth=0.0,         # 单卡, 不 inter-node
        inter_node_latency=0.0,
        # comm_plan Step 2: 新通信参数结构. AllReduce 已统一走 NCCL candidate 模型;
        # comm_step_latency / intra_node_protocol_efficiency 仍供其它 legacy collective 使用.
        communication=rtx_4090_nccl_communication(),
    ),
}


PROFILE_ALIASES: dict[str, str] = {
    "nvidia_A100": "A100",
    "nvidia_H100": "H100",
    "nvidia_H800": "H800",
    "nvidia_H200": "H200",
    "nvidia_H20_96G": "H20_96G",
    "nvidia_B200": "B200",
    "nvidia_B300": "B300",
    "Ascend_950PR": "Ascend_950PR",
    "Ascend_950DT": "Ascend_950DT",
    "ascend_950pr": "Ascend_950PR",
    "ascend_950dt": "Ascend_950DT",
    "nvidia_RTX_4090": "RTX_4090",
    "rtx_4090": "RTX_4090",
    "RTX4090": "RTX_4090",
    "rtx4090": "RTX_4090",
}


def get_hardware_config(name: str, calibrated: bool = True) -> HardwareConfig:
    """Get a flat HardwareConfig by canonical name or compatibility alias."""
    canonical = PROFILE_ALIASES.get(name, name)
    if canonical not in KNOWN_PROFILES:
        known = sorted([*KNOWN_PROFILES.keys(), *PROFILE_ALIASES.keys()])
        raise KeyError(f"Unknown hardware: {name}. Known: {known}")
    params = dict(KNOWN_PROFILES[canonical])
    if not calibrated:
        params["compute_efficiency"] = 1.0
        params["mem_efficiency"] = 1.0
        params["comm_efficiency"] = 1.0
    return HardwareConfig(name=name, **params)


def get_hardware_profile(name: str, calibrated: bool = True) -> HardwareProfile:
    """Get a structured HardwareProfile (scenario 域入口)."""
    return HardwareProfile.from_legacy(get_hardware_config(name, calibrated=calibrated))


def get_hardware_perf_table_params(name: str) -> dict:
    """Return flat hardware params using the k8s-deploy perf table field names."""
    canonical = PROFILE_ALIASES.get(name, name)
    if canonical not in KNOWN_PROFILES:
        known = sorted([*KNOWN_PROFILES.keys(), *PROFILE_ALIASES.keys()])
        raise KeyError(f"Unknown hardware: {name}. Known: {known}")
    p = KNOWN_PROFILES[canonical]

    def precision_value(key: str):
        field = {
            "FP16": "peak_flops_fp16",
            "BF16": "peak_flops_bf16",
            "INT8": "peak_flops_int8",
            "FP8": "peak_flops_fp8",
            "FP4": "peak_flops_fp4",
        }[key]
        value = p.get(field, 0.0)
        if value == 0.0 and key in {"FP8", "FP4"}:
            return "N/A"
        return value

    return {
        "bandwidth": p.get("mem_bandwidth", 0.0),
        "HBM_capacity_GB": p.get("mem_capacity_gb", 0.0),
        "BF16": precision_value("BF16"),
        "FP16": precision_value("FP16"),
        "FP8": precision_value("FP8"),
        "INT8": precision_value("INT8"),
        "FP4": precision_value("FP4"),
        "vector_flops": p.get("vector_flops", p.get("peak_flops_fp16", 0.0)),
        "interconnect_bandwidth": p.get("intra_node_bandwidth", 0.0) / 1e9,
        "onchip_buffer": p.get("onchip_buffer", 0.0),
    }
