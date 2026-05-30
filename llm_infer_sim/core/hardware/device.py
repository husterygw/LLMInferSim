"""Hardware device config (config_plan §4.4)。

承载两类对象:
  - 扁平 `HardwareConfig`：canonical 物理能力 + effective_* / 拓扑缩放计算逻辑
    (从 legacy core/profiles/hardware.py 原样迁入, 数值零漂移)。cost backend /
    roofline 公式直读它的 effective_* 属性 —— 作为 compat 边界保留。
  - 结构化 `HardwareProfile` (+ ComputeSpec/MemorySpec)：scenario 域聚合, 组合
    结构化 spec, 通过 to_legacy() round-trip 回 HardwareConfig 复用其计算。

效率系数 (compute/mem/comm_efficiency) 物理上仍随硬件携带 (嵌入 effective_*)。
校准 knob (kernel_overhead / moe_efficiency) 已迁出 (config_plan Step G) 到
CalibrationProfile, 不再 masquerade 成硬件物理 spec。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Optional

from llm_infer_sim.core.hardware.communication import CommunicationProfile
from llm_infer_sim.core.hardware.topology import NodeTopology


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

    # comm_plan Step 2: per-collective 通信参数 (替代 comm_step_latency 这种"万能 alpha").
    # 默认 None, 由 profile entry 显式注入. Step 4 接通 RooflineBackend 后取代旧字段.
    communication: Optional[CommunicationProfile] = None

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


@dataclass(frozen=True)
class ComputeSpec:
    peak_flops_fp16: float = 989e12
    peak_flops_bf16: float = 0.0
    peak_flops_int8: float = 0.0
    peak_flops_fp8: float = 0.0
    peak_flops_fp4: float = 0.0
    has_fp4_tc: bool = False
    vector_flops: float = 0.0


@dataclass(frozen=True)
class MemorySpec:
    mem_bandwidth: float = 3.35e12
    mem_capacity_gb: float = 80.0
    onchip_buffer: float = 33792e3


@dataclass(frozen=True)
class HardwareProfile:
    name: str = "custom"
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    memory: MemorySpec = field(default_factory=MemorySpec)
    topology: NodeTopology = field(default_factory=NodeTopology)
    communication: Optional[CommunicationProfile] = None

    # 效率/校准 + 杂项扩展 (round-trip 精确; 不改数值口径)
    compute_efficiency: float = 1.0
    mem_efficiency: float = 1.0
    comm_efficiency: float = 1.0
    optional_collective_floor: Optional[dict] = None
    framework_call_overhead: Optional[dict] = None
    collective_algo_bias: Optional[dict] = None

    @classmethod
    def from_legacy(cls, hw: HardwareConfig) -> "HardwareProfile":
        return cls(
            name=hw.name,
            compute=ComputeSpec(
                peak_flops_fp16=hw.peak_flops_fp16,
                peak_flops_bf16=hw.peak_flops_bf16,
                peak_flops_int8=hw.peak_flops_int8,
                peak_flops_fp8=hw.peak_flops_fp8,
                peak_flops_fp4=hw.peak_flops_fp4,
                has_fp4_tc=hw.has_fp4_tc,
                vector_flops=hw.vector_flops,
            ),
            memory=MemorySpec(
                mem_bandwidth=hw.mem_bandwidth,
                mem_capacity_gb=hw.mem_capacity_gb,
                onchip_buffer=hw.onchip_buffer,
            ),
            topology=NodeTopology(
                intra_node_bandwidth=hw.intra_node_bandwidth,
                intra_node_size=hw.intra_node_size,
                intra_node_topology=hw.intra_node_topology,
                intra_node_protocol_efficiency=hw.intra_node_protocol_efficiency,
                intra_node_gpus_per_root=hw.intra_node_gpus_per_root,
                intra_node_num_roots=hw.intra_node_num_roots,
                gpu_to_root=hw.gpu_to_root,
                comm_step_latency=hw.comm_step_latency,
                has_nvlink_sharp=hw.has_nvlink_sharp,
                enable_nvls_model=hw.enable_nvls_model,
                inter_node_bandwidth=hw.inter_node_bandwidth,
                inter_node_latency=hw.inter_node_latency,
            ),
            communication=hw.communication,
            compute_efficiency=hw.compute_efficiency,
            mem_efficiency=hw.mem_efficiency,
            comm_efficiency=hw.comm_efficiency,
            optional_collective_floor=hw.optional_collective_floor,
            framework_call_overhead=hw.framework_call_overhead,
            collective_algo_bias=hw.collective_algo_bias,
        )

    def to_legacy(self) -> HardwareConfig:
        c, m, t = self.compute, self.memory, self.topology
        return HardwareConfig(
            name=self.name,
            peak_flops_fp16=c.peak_flops_fp16,
            peak_flops_bf16=c.peak_flops_bf16,
            peak_flops_int8=c.peak_flops_int8,
            peak_flops_fp8=c.peak_flops_fp8,
            peak_flops_fp4=c.peak_flops_fp4,
            has_fp4_tc=c.has_fp4_tc,
            vector_flops=c.vector_flops,
            mem_bandwidth=m.mem_bandwidth,
            mem_capacity_gb=m.mem_capacity_gb,
            onchip_buffer=m.onchip_buffer,
            intra_node_bandwidth=t.intra_node_bandwidth,
            intra_node_size=t.intra_node_size,
            intra_node_topology=t.intra_node_topology,
            intra_node_protocol_efficiency=t.intra_node_protocol_efficiency,
            intra_node_gpus_per_root=t.intra_node_gpus_per_root,
            intra_node_num_roots=t.intra_node_num_roots,
            gpu_to_root=t.gpu_to_root,
            comm_step_latency=t.comm_step_latency,
            has_nvlink_sharp=t.has_nvlink_sharp,
            enable_nvls_model=t.enable_nvls_model,
            inter_node_bandwidth=t.inter_node_bandwidth,
            inter_node_latency=t.inter_node_latency,
            compute_efficiency=self.compute_efficiency,
            mem_efficiency=self.mem_efficiency,
            comm_efficiency=self.comm_efficiency,
            communication=self.communication,
            optional_collective_floor=self.optional_collective_floor,
            framework_call_overhead=self.framework_call_overhead,
            collective_algo_bias=self.collective_algo_bias,
        )

    # ---- flat read facade (config_plan Step E) ----
    # 让 RooflineBackend / RooflineAnalyzer / communication.py 直读结构化 HardwareProfile,
    # 无需 to_legacy() 重建 HardwareConfig。每个属性/方法逐式复刻 HardwareConfig 的
    # __post_init__ 派生 + effective_* / 拓扑缩放公式, 故对 cost 路径 byte-identical。
    # HardwareConfig 与 HardwareProfile 在这些读法上 duck-type 等价。

    @property
    def _peak_fp16(self) -> float:
        return self.compute.peak_flops_fp16

    @property
    def _peak_bf16(self) -> float:
        c = self.compute
        return c.peak_flops_bf16 or c.peak_flops_fp16

    @property
    def _peak_int8(self) -> float:
        c = self.compute
        return c.peak_flops_int8 or c.peak_flops_fp16 * 2

    @property
    def _peak_fp8(self) -> float:
        c = self.compute
        return c.peak_flops_fp8 or self._peak_int8

    @property
    def _peak_fp4(self) -> float:
        c = self.compute
        if c.peak_flops_fp4:
            return c.peak_flops_fp4
        return c.peak_flops_fp16 * 4 if c.has_fp4_tc else 0.0

    @property
    def _vector_flops(self) -> float:
        c = self.compute
        return c.vector_flops or c.peak_flops_fp16

    @property
    def has_fp4_tc(self) -> bool:
        return self.compute.has_fp4_tc

    @property
    def effective_peak_flops(self) -> float:
        return self._peak_fp16 * self.compute_efficiency

    @property
    def effective_peak_int8(self) -> float:
        return self._peak_int8 * self.compute_efficiency

    @property
    def effective_peak_bf16(self) -> float:
        return self._peak_bf16 * self.compute_efficiency

    @property
    def effective_peak_fp8(self) -> float:
        return self._peak_fp8 * self.compute_efficiency

    @property
    def effective_peak_fp4(self) -> float:
        return self._peak_fp4 * self.compute_efficiency

    @property
    def effective_vector_flops(self) -> float:
        return self._vector_flops * self.compute_efficiency

    @property
    def effective_mem_bandwidth(self) -> float:
        return self.memory.mem_bandwidth * self.mem_efficiency

    # 内存物理量 (attention/mla FA tile sizing + worker KV budget 直读)
    @property
    def onchip_buffer(self) -> float:
        return self.memory.onchip_buffer

    @property
    def mem_capacity_gb(self) -> float:
        return self.memory.mem_capacity_gb

    @property
    def mem_bandwidth(self) -> float:
        return self.memory.mem_bandwidth

    @property
    def effective_comm_bandwidth(self) -> float:
        return self.topology.intra_node_bandwidth * self.comm_efficiency / 2

    # 拓扑物理量 (communication.py 直读)
    @property
    def intra_node_size(self) -> int:
        return self.topology.intra_node_size

    @property
    def intra_node_protocol_efficiency(self) -> float:
        return self.topology.intra_node_protocol_efficiency

    @property
    def comm_step_latency(self) -> float:
        return self.topology.comm_step_latency

    @property
    def inter_node_latency(self) -> float:
        return self.topology.inter_node_latency

    @property
    def has_nvlink_sharp(self) -> bool:
        return self.topology.has_nvlink_sharp

    @property
    def enable_nvls_model(self) -> bool:
        return self.topology.enable_nvls_model

    def effective_intra_bw(
        self,
        n: int = 1,
        topology_hint: TopologyHint = "concentrated",
        visible_devices: Optional[list] = None,
    ) -> float:
        t = self.topology
        raw = (
            t.intra_node_bandwidth / 2
            * self.comm_efficiency
            * t.intra_node_protocol_efficiency
        )
        if t.intra_node_topology == "pcie_shared_root":
            n_per_root = self._estimate_n_per_root(
                n=n,
                topology_hint=topology_hint,
                visible_devices=visible_devices,
            )
            return raw / max(n_per_root, 1)
        return raw

    def _estimate_n_per_root(
        self,
        n: int,
        topology_hint: TopologyHint = "concentrated",
        visible_devices: Optional[list] = None,
    ) -> int:
        t = self.topology
        if visible_devices:
            roots = []
            gps_per_root = max(t.intra_node_gpus_per_root, 1)
            for gpu_id in visible_devices[:n]:
                if t.gpu_to_root is not None:
                    root = t.gpu_to_root.get(gpu_id, gpu_id // gps_per_root)
                else:
                    root = gpu_id // gps_per_root
                roots.append(root)
            counts = Counter(roots)
            return max(counts.values()) if counts else 1
        if topology_hint == "balanced":
            return math.ceil(n / max(t.intra_node_num_roots, 1))
        return min(n, t.intra_node_gpus_per_root)

    @property
    def effective_inter_bw(self) -> float:
        return self.topology.inter_node_bandwidth * self.comm_efficiency

    @property
    def ridge_point(self) -> float:
        if self.effective_mem_bandwidth == 0:
            return float("inf")
        return self.effective_peak_flops / self.effective_mem_bandwidth
