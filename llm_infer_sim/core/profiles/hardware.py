"""Hardware config + profiles (merged from llm-viewer hardware/{config,profiles}.py).

Source: llm-viewer/hardware/config.py + llm-viewer/hardware/profiles.py
Strategy: 复制 + 扩展 (详设 §2 表)。"""
"""Hardware configuration dataclass.

Extends the original llm-viewer flat dict with efficiency calibration,
vector FLOPS, interconnect bandwidth, and kernel overhead from llm_inference_eval.
Also provides conversion to/from the legacy flat-dict format used by
hardwares/hardware_params.py and the frontend.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Optional


TopologyHint = Literal["concentrated", "balanced"]


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

    # --- Interconnect: intra-node (NVLink / PCIe) ---
    # 详细设计见 docs/COMMUNICATION_MODELING.md (Phase 5).
    intra_node_bandwidth: float = 450e9  # bidirectional B/s (nominal spec convention)
    intra_node_size: int = 8             # GPUs per node (standard NVIDIA HGX = 8)

    # comm_step_latency: NCCL collective 每个逻辑通信 step 的 effective latency。
    # 包含: 链路访问 + GPU 间路由 + NCCL step 内部同步 + 小包协议开销。
    # **不是裸 PCIe/NVLink SerDes latency**。推荐:
    #   NVLink/NVSwitch: 1e-6
    #   PCIe 4.0/5.0:    5e-6
    # 重命名自旧字段 link_latency (Phase 5 起). 所有 profile dict 用新名,
    # callers (communication.py / tests) 已同步更新到 hw.comm_step_latency.
    comm_step_latency: float = 1e-6

    # 拓扑类型, 决定 AllReduce 时 effective β 怎么 scale:
    #   "nvlink_full"      — 每对 GPU 独占链路, β 与 n 无关
    #   "pcie_shared_root" — n 张卡共享 PCIe root, β 按 n_per_root 缩
    intra_node_topology: str = "nvlink_full"

    # NCCL 协议层效率: nominal busbw / (raw_bandwidth/2). 典型 0.55-0.8.
    # 来源: NCCL 协议控制 + chunking + sync + encoding 综合折扣。
    intra_node_protocol_efficiency: float = 0.7

    # --- PCIe topology hint (only used when topology="pcie_shared_root") ---
    intra_node_gpus_per_root: int = 8     # PCIe 单 root 最多挂几张卡
    intra_node_num_roots: int = 1         # 该 server 上 PCIe root 个数
    # 可选: 显式 GPU id -> root id 映射. None 则 fallback 到 gpu_id // gpus_per_root.
    gpu_to_root: Optional[dict] = None

    # --- NCCL capability ---
    has_nvlink_sharp: bool = False         # 硬件能力 (H100/H200/B200 SXM = True)
    # 是否在 AllReduce 候选算法中加入 NVLS 公式. 默认 False, 校准后再启用。
    # 详 docs/COMMUNICATION_MODELING.md §14.
    enable_nvls_model: bool = False

    # --- Optional extensions (default disabled) ---
    # 通信小消息 floor (per-collective dict). 默认 None, 不参与计算.
    # 用于 H100/B200/inter-node 等场景下 algorithm_term(data→0) 系统性低估的 fallback.
    optional_collective_floor: Optional[dict] = None
    # framework_call_overhead override (per-collective dict). 默认 None,
    # 使用 communication.DEFAULT_FRAMEWORK_CALL_OVERHEAD.
    framework_call_overhead: Optional[dict] = None
    # 算法 bias: {coll: {algo: factor}}. 默认 None, 所有 bias = 1.0.
    # 用于校正 min(candidate) 跟 NCCL 真实选择的偏差.
    collective_algo_bias: Optional[dict] = None

    # --- Interconnect: inter-node (InfiniBand / Ethernet) ---
    inter_node_bandwidth: float = 50e9   # unidirectional B/s (IB spec convention)
                                         # NDR400: 400 Gbps ≈ 50 GB/s per port (default)
    inter_node_latency: float = 5e-6

    # --- Efficiency calibration (0.0 ~ 1.0) ---
    compute_efficiency: float = 1.0      # default 1.0 = no calibration (Phase 1 compat)
    mem_efficiency: float = 1.0
    comm_efficiency: float = 1.0

    # 2026-05-17 baseline cleanup: 默认 {} (空, 不加 per-op overhead).
    # 旧值 {"default": 2e-6} 是从 RTX 4090 calibration α intercept 反推,
    # 含义混杂 (GPU kernel start floor + 隐式 framework overhead). 违反"独立实测"原则.
    # Stage A 校准时若发现需要, 走 scripts/measure_compute_overhead.py (待写) 独测.
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
        # intra_node_bandwidth 是双向带宽, 除 2 得到单向。
        # 这是 raw 物理 BW (含 comm_efficiency 校准), 不含 NCCL 协议层 / topology 拓扑修正。
        # 阶段 7+ 公式优先用 effective_intra_bw(n) 方法 (拓扑感知), 本属性留作 legacy fallback。
        return self.intra_node_bandwidth * self.comm_efficiency / 2

    def effective_intra_bw(
        self,
        n: int = 1,
        topology_hint: TopologyHint = "concentrated",
        visible_devices: Optional[list] = None,
    ) -> float:
        """Ring/Tree collective 公式可直接使用的 per-link effective β (单向 B/s)。

        Args:
            n: 参与 collective 的 GPU 数。
            topology_hint: 当 visible_devices 未给时,选择拓扑分布假设:
                "concentrated" — n 张卡都在同一个 PCIe root(保守 / 默认)
                "balanced"     — n 张卡均匀分布在所有 num_roots 个 root 上
            visible_devices: 实际可见 GPU id 列表 (e.g. [0,1,4,5]). 给定时优先用
                它通过 gpu_to_root (或 fallback gpu_id // gpus_per_root) 推 n_per_root。

        β = (intra_node_bandwidth / 2) × comm_efficiency
          × intra_node_protocol_efficiency × topology_factor(n)

        topology_factor:
            "nvlink_full":      1                (每对 GPU 独占链路, 不竞争)
            "pcie_shared_root": 1 / n_per_root   (按 root contention 缩)
        """
        raw = (
            self.intra_node_bandwidth / 2
            * self.comm_efficiency
            * self.intra_node_protocol_efficiency
        )
        if self.intra_node_topology == "pcie_shared_root":
            n_per_root = self._estimate_n_per_root(
                n=n,
                topology_hint=topology_hint,
                visible_devices=visible_devices,
            )
            return raw / max(n_per_root, 1)
        # nvlink_full / default 不缩
        return raw

    def _estimate_n_per_root(
        self,
        n: int,
        topology_hint: TopologyHint = "concentrated",
        visible_devices: Optional[list] = None,
    ) -> int:
        """估算 ring/tree 中 single root 上同时通信的 GPU 数 (用于 β 缩放)."""
        if visible_devices:
            roots = []
            gps_per_root = max(self.intra_node_gpus_per_root, 1)
            for gpu_id in visible_devices[:n]:
                if self.gpu_to_root is not None:
                    root = self.gpu_to_root.get(gpu_id, gpu_id // gps_per_root)
                else:
                    root = gpu_id // gps_per_root
                roots.append(root)
            counts = Counter(roots)
            return max(counts.values()) if counts else 1
        # 没 visible_devices, 按 hint
        if topology_hint == "balanced":
            return math.ceil(n / max(self.intra_node_num_roots, 1))
        # concentrated (默认): 假设 n 张卡都在同一 root, 但不超过 root 容量
        return min(n, self.intra_node_gpus_per_root)


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
        onchip_buffer=72 * 1024 * 1024,   # L2 cache 72 MB
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
