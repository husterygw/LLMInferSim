"""PDDisaggConfig — Prefill-Decode 分离配置 (config_plan §4.3 / 详设 §7.6)。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# PD 分离 connector → 默认带宽 preset (GB/s + 起始 latency us).
# 名字与 vLLM 0.20.1 KVConnectorFactory.register_connector 完全一致.
# - P2pNcclConnector: 同集群 GPU NCCL P2P, 走 IB/NVLink, ~25-50 GB/s
# - LMCacheConnectorV1 / LMCacheMPConnector: 走 RDMA 经 CPU, ~12 GB/s
# - MooncakeConnector: RDMA 直传, ~25 GB/s
# - NixlConnector: NVIDIA NIXL 新方案, 类 NCCL
# - OffloadingConnector / SimpleCPUOffloadConnector: 走 CPU mem, ~8 GB/s (host RAM)
# - HF3FSKVConnector: HuggingFace 3FS 分布式 fs
# 实测值可在 hardware preset 里 override; 这里仅作 fallback。
PD_CONNECTOR_PRESETS: dict[str, tuple[float, float]] = {
    # connector_name → (bandwidth_GBps, startup_latency_us)
    "P2pNcclConnector":          (25.0, 5.0),
    "LMCacheConnectorV1":        (12.0, 100.0),
    "LMCacheMPConnector":        (12.0, 100.0),
    "MooncakeConnector":         (25.0, 30.0),
    "NixlConnector":             (25.0, 5.0),
    "OffloadingConnector":       (8.0, 50.0),
    "SimpleCPUOffloadConnector": (8.0, 50.0),
    "MultiConnector":            (15.0, 50.0),  # 复合, 走最慢通道近似
    "FlexKVConnectorV1":         (15.0, 50.0),
    "MoRIIOConnector":           (15.0, 50.0),
    "HF3FSKVConnector":          (3.0, 1000.0),  # 分布式 fs, 慢但容量大
}


@dataclass
class PDDisaggConfig:
    """Prefill-Decode 分离配置 (详设 §7.6).

    None / role==None 表示未启用 PD 分离。启用时:
      - role=kv_producer: 当前进程是 prefill node, 在 prefill 完成时 send KV
      - role=kv_consumer: 当前进程是 decode node, 在首次 decode 之前等 recv KV
      - role=kv_both: 兼任 producer + consumer (同进程跑 prefill+decode)

    connector_bandwidth_gbps / connector_latency_us 若设, 优先于 hardware
    preset; 否则按 connector_name 从 PD_CONNECTOR_PRESETS 取。
    """
    role: Literal["kv_producer", "kv_consumer", "kv_both"] | None = None
    connector_name: str | None = None     # 如 "P2pNcclConnector"
    kv_parallel_size: int = 1
    connector_bandwidth_gbps: float | None = None
    connector_latency_us: float | None = None

    @property
    def enabled(self) -> bool:
        return self.role is not None

    @property
    def is_producer(self) -> bool:
        return self.role in ("kv_producer", "kv_both")

    @property
    def is_consumer(self) -> bool:
        return self.role in ("kv_consumer", "kv_both")

    def resolve_bandwidth(self) -> float:
        """resolve GB/s, 显式 > preset > fallback 10 GB/s."""
        if self.connector_bandwidth_gbps is not None:
            return self.connector_bandwidth_gbps
        if self.connector_name in PD_CONNECTOR_PRESETS:
            return PD_CONNECTOR_PRESETS[self.connector_name][0]
        return 10.0

    def resolve_latency_us(self) -> float:
        """resolve startup latency (us)."""
        if self.connector_latency_us is not None:
            return self.connector_latency_us
        if self.connector_name in PD_CONNECTOR_PRESETS:
            return PD_CONNECTOR_PRESETS[self.connector_name][1]
        return 50.0


__all__ = ["PDDisaggConfig", "PD_CONNECTOR_PRESETS"]
