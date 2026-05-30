"""NodeTopology — 节点内/间互联拓扑 (config_plan §4.4)。

物理互联描述 (intra/inter-node 带宽、拓扑类型、PCIe root 映射、协议效率、
NCCL 能力)。供 HardwareProfile 组合; effective β / 拓扑缩放逻辑保留在
HardwareProfile (round-trip 到 legacy HardwareConfig 计算)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NodeTopology:
    # intra-node (NVLink / PCIe)
    intra_node_bandwidth: float = 450e9
    intra_node_size: int = 8
    intra_node_topology: str = "nvlink_full"
    intra_node_protocol_efficiency: float = 0.7
    intra_node_gpus_per_root: int = 8
    intra_node_num_roots: int = 1
    gpu_to_root: Optional[dict] = None
    comm_step_latency: float = 1e-6
    # NCCL capability
    has_nvlink_sharp: bool = False
    enable_nvls_model: bool = False
    # inter-node (InfiniBand / Ethernet)
    inter_node_bandwidth: float = 50e9
    inter_node_latency: float = 5e-6
