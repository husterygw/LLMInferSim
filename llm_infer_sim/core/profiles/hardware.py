"""Hardware config + profiles (merged from llm-viewer hardware/{config,profiles}.py).

Source: llm-viewer/hardware/config.py + llm-viewer/hardware/profiles.py
Strategy: 复制 + 扩展 (详设 §2 表)。"""
"""Hardware configuration dataclass.

Extends the original llm-viewer flat dict with efficiency calibration,
vector FLOPS, interconnect bandwidth, and kernel overhead from llm_inference_eval.
Also provides conversion to/from the legacy flat-dict format used by
hardwares/hardware_params.py and the frontend.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareConfig:
    """Complete hardware accelerator description."""

    name: str = "custom"

    # --- Compute ---
    peak_flops_fp16: float = 989e12      # FP16 Tensor Core peak OPS
    peak_flops_bf16: float = 0.0         # BF16 Tensor Core peak OPS (0 = same as FP16)
    peak_flops_int8: float = 0.0         # INT8 peak OPS (0 = auto 2x fp16)
    peak_flops_fp8: float = 0.0          # FP8 Tensor Core peak OPS (0 = auto 2x fp16)
    peak_flops_fp4: float = 0.0          # FP4 Tensor Core peak OPS (0 = auto 4x fp16 if has_fp4_tc)
    has_fp4_tc: bool = False             # Has native FP4 Tensor Core (Blackwell+)
    vector_flops: float = 0.0            # CUDA Core peak (Norm/Activation); 0 = same as fp16

    # --- Memory ---
    mem_bandwidth: float = 3.35e12       # HBM bandwidth B/s
    mem_capacity_gb: float = 80.0        # HBM capacity GB
    onchip_buffer: float = 33792e3       # On-chip SRAM bytes (FlashAttention tile sizing)

    # --- Interconnect: intra-node (NVLink) ---
    intra_node_bandwidth: float = 450e9  # bidirectional B/s (NVLink spec convention)
    intra_node_size: int = 8             # GPUs per node (standard NVIDIA HGX = 8)
    link_latency: float = 1e-6           # seconds

    # --- Interconnect: inter-node (InfiniBand / Ethernet) ---
    inter_node_bandwidth: float = 50e9   # unidirectional B/s (IB spec convention)
                                         # NDR400: 400 Gbps ≈ 50 GB/s per port (default)
    inter_node_latency: float = 5e-6

    # --- Efficiency calibration (0.0 ~ 1.0) ---
    compute_efficiency: float = 1.0      # default 1.0 = no calibration (Phase 1 compat)
    mem_efficiency: float = 1.0
    comm_efficiency: float = 1.0

    # --- Kernel overhead (seconds, keyed by op_category) ---
    kernel_overhead: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.peak_flops_bf16 == 0.0:
            self.peak_flops_bf16 = self.peak_flops_fp16
        if self.peak_flops_int8 == 0.0:
            self.peak_flops_int8 = self.peak_flops_fp16 * 2
        if self.peak_flops_fp8 == 0.0:
            self.peak_flops_fp8 = self.peak_flops_int8
        if self.peak_flops_fp4 == 0.0 and self.has_fp4_tc:
            self.peak_flops_fp4 = self.peak_flops_fp16 * 4
        if self.vector_flops == 0.0:
            self.vector_flops = self.peak_flops_fp16

    # --- Effective values (raw × efficiency) ---

    @property
    def effective_peak_flops(self) -> float:
        return self.peak_flops_fp16 * self.compute_efficiency

    @property
    def effective_peak_int8(self) -> float:
        return self.peak_flops_int8 * self.compute_efficiency

    @property
    def effective_peak_bf16(self) -> float:
        return self.peak_flops_bf16 * self.compute_efficiency

    @property
    def effective_peak_fp8(self) -> float:
        return self.peak_flops_fp8 * self.compute_efficiency

    @property
    def effective_peak_fp4(self) -> float:
        return self.peak_flops_fp4 * self.compute_efficiency

    @property
    def effective_vector_flops(self) -> float:
        return self.vector_flops * self.compute_efficiency

    @property
    def effective_mem_bandwidth(self) -> float:
        return self.mem_bandwidth * self.mem_efficiency

    @property
    def effective_comm_bandwidth(self) -> float:
        # intra_node_bandwidth是双向带宽，这里需要除以2来得到单向带宽
        # 阶段 7 起公式有 hierarchical 路径优先用 effective_intra_bw / effective_inter_bw,
        # 本属性保留供阶段 0-6 旧路径 (N ≤ intra_node_size) fallback 用。
        return self.intra_node_bandwidth * self.comm_efficiency / 2

    @property
    def effective_intra_bw(self) -> float:
        """单向 intra-node 带宽 (跟 effective_comm_bandwidth 同, 显式命名供阶段 7 公式用)."""
        return self.intra_node_bandwidth * self.comm_efficiency / 2

    @property
    def effective_inter_bw(self) -> float:
        """单向 inter-node 带宽 (IB / Ethernet)。

        IB spec 通常直接以单向为口径 (NDR400 = 50 GB/s per port unidirectional),
        所以不再除 2。返回 0 时阶段 7 公式 fallback 到 flat ring 全 intra_bw。
        """
        return self.inter_node_bandwidth * self.comm_efficiency

    @property
    def ridge_point(self) -> float:
        """Roofline turning point = peak / bandwidth (FLOP/byte)."""
        if self.effective_mem_bandwidth == 0:
            return float("inf")
        return self.effective_peak_flops / self.effective_mem_bandwidth

# ===== profiles (merged from hardware/profiles.py) =====
"""Pre-defined hardware profiles.

Values are converted from origin/k8s-deploy:hardwares/hardware_params.py
into the local HardwareConfig format. Numeric expressions intentionally
stay close to the source hardware_params.py values; only field names and
interconnect bandwidth units are adapted for HardwareConfig.
"""



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
        link_latency=0.0,
        inter_node_bandwidth=2.5e+10,
        inter_node_latency=0.0,
    ),
    "A100_40G": dict(
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
        link_latency=0.0,
        inter_node_bandwidth=2.5e+10,
        inter_node_latency=0.0,
    ),
    "A100_80G": dict(
        peak_flops_fp16=312e12,
        peak_flops_bf16=312e12,
        peak_flops_int8=624e12,
        peak_flops_fp8=0.0,
        peak_flops_fp4=0.0,
        vector_flops=19.5e12,
        has_fp4_tc=False,
        mem_bandwidth=2039e9,
        mem_capacity_gb=80,
        onchip_buffer=27648e3,
        intra_node_bandwidth=600e9,
        link_latency=0.0,
        inter_node_bandwidth=2.5e+10,
        inter_node_latency=0.0,
    ),
    "A800_80G_SXM": dict(
        peak_flops_fp16=312e12,
        peak_flops_bf16=312e12,
        peak_flops_int8=624e12,
        peak_flops_fp8=0.0,
        peak_flops_fp4=0.0,
        vector_flops=19.5e12,
        has_fp4_tc=False,
        mem_bandwidth=2039e9,
        mem_capacity_gb=80,
        onchip_buffer=27648e3,
        intra_node_bandwidth=400e9,
        link_latency=0.0,
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
        link_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H100_SXM": dict(
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
        link_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H100_SXM_Sparse": dict(
        peak_flops_fp16=1979e12,
        peak_flops_bf16=1979e12,
        peak_flops_int8=3958e12,
        peak_flops_fp8=3958e12,
        peak_flops_fp4=0.0,
        vector_flops=67e12,
        has_fp4_tc=False,
        mem_bandwidth=3350e9,
        mem_capacity_gb=80,
        onchip_buffer=33792e3,
        intra_node_bandwidth=900e9,
        link_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H100_PCIe": dict(
        peak_flops_fp16=1513e12 / 2,
        peak_flops_bf16=1513e12 / 2,
        peak_flops_int8=3026e12 / 2,
        peak_flops_fp8=3026e12 / 2,
        peak_flops_fp4=0.0,
        vector_flops=51e12,
        has_fp4_tc=False,
        mem_bandwidth=2048e9,
        mem_capacity_gb=80,
        onchip_buffer=29184e3,
        intra_node_bandwidth=600e9,
        link_latency=0.0,
        inter_node_bandwidth=2.5e+10,
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
        link_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H800_SXM": dict(
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
        link_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H800_PCIe": dict(
        peak_flops_fp16=1513e12 / 2,
        peak_flops_bf16=1513e12 / 2,
        peak_flops_int8=3026e12 / 2,
        peak_flops_fp8=3026e12 / 2,
        peak_flops_fp4=0.0,
        vector_flops=51e12,
        has_fp4_tc=False,
        mem_bandwidth=2000e9,
        mem_capacity_gb=80,
        onchip_buffer=29184e3,
        intra_node_bandwidth=400e9,
        link_latency=0.0,
        inter_node_bandwidth=2.5e+10,
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
        link_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H200_SXM": dict(
        peak_flops_fp16=1979e12 / 2,
        peak_flops_bf16=1979e12 / 2,
        peak_flops_int8=3958e12 / 2,
        peak_flops_fp8=3958e12 / 2,
        peak_flops_fp4=0.0,
        vector_flops=67e12,
        has_fp4_tc=False,
        mem_bandwidth=4800e9,
        mem_capacity_gb=141,
        onchip_buffer=33792e3,
        intra_node_bandwidth=900e9,
        link_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H200_SXM_Sparse": dict(
        peak_flops_fp16=1979e12,
        peak_flops_bf16=1979e12,
        peak_flops_int8=3958e12,
        peak_flops_fp8=3958e12,
        peak_flops_fp4=0.0,
        vector_flops=67e12,
        has_fp4_tc=False,
        mem_bandwidth=4800e9,
        mem_capacity_gb=141,
        onchip_buffer=33792e3,
        intra_node_bandwidth=900e9,
        link_latency=0.0,
        inter_node_bandwidth=5e+10,
        inter_node_latency=0.0,
    ),
    "H200_NVL": dict(
        peak_flops_fp16=1671e12,
        peak_flops_bf16=1671e12,
        peak_flops_int8=3341e12,
        peak_flops_fp8=3341e12,
        peak_flops_fp4=0.0,
        vector_flops=60e12,
        has_fp4_tc=False,
        mem_bandwidth=4800e9,
        mem_capacity_gb=141,
        onchip_buffer=33792e3,
        intra_node_bandwidth=900e9,
        link_latency=0.0,
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
        link_latency=0.0,
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
        link_latency=0.0,
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
        link_latency=0.0,
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
        link_latency=0.0,
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
        link_latency=0.0,
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
        link_latency=0.0,
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
        link_latency=0.0,
        inter_node_bandwidth=0.0,
        inter_node_latency=0.0,
    ),
}


PROFILE_ALIASES: dict[str, str] = {
    "nvidia_A100": "A100",
    "nvidia_A100_40G": "A100_40G",
    "nvidia_A100_80G": "A100_80G",
    "nvidia_A800_80G_SXM": "A800_80G_SXM",
    "nvidia_H100": "H100",
    "nvidia_H100_SXM": "H100_SXM",
    "nvidia_H100_SXM_Sparse": "H100_SXM_Sparse",
    "nvidia_H100_PCIe": "H100_PCIe",
    "nvidia_H800": "H800",
    "nvidia_H800_SXM": "H800_SXM",
    "nvidia_H800_PCIe": "H800_PCIe",
    "nvidia_H200": "H200",
    "nvidia_H200_SXM": "H200_SXM",
    "nvidia_H200_SXM_Sparse": "H200_SXM_Sparse",
    "nvidia_H200_NVL": "H200_NVL",
    "nvidia_H20_96G": "H20_96G",
    "nvidia_B200": "B200",
    "nvidia_B300": "B300",
    "nvidia_Ascend_950PR": "Ascend_950PR",
    "nvidia_Ascend_950DT": "Ascend_950DT",
    "H200_141GB_SXM": "H200_SXM",
    "nvidia_H200_141GB_SXM": "H200_SXM",
    "ascend_950pr": "Ascend_950PR",
    "ascend_950dt": "Ascend_950DT",
}


def get_hardware_profile(name: str, calibrated: bool = True) -> HardwareConfig:
    """Get a hardware profile by canonical name or compatibility alias."""
    canonical = PROFILE_ALIASES.get(name, name)
    if canonical not in KNOWN_PROFILES:
        known = sorted([*KNOWN_PROFILES.keys(), *PROFILE_ALIASES.keys()])
        raise KeyError(f"Unknown hardware: {name}. Known: {known}")
    params = dict(KNOWN_PROFILES[canonical])
    if not calibrated:
        params["compute_efficiency"] = 1.0
        params["mem_efficiency"] = 1.0
        params["comm_efficiency"] = 1.0
        params["kernel_overhead"] = {}
    return HardwareConfig(name=name, **params)


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
