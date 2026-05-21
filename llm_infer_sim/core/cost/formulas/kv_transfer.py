"""KV cache 跨节点传输 cost (PD 分离 §7.6).

vLLM PD 分离: prefill 完成后, 该 sequence 的全部 KV block 从 producer 节点 send 到
consumer 节点; consumer 接到后才开始 decode。

cost 简单线性模型:
    time = startup_latency + bytes / bandwidth

bytes 来自 KVBlockAllocator.req_kv_bytes(req_id) (block-aligned, 含全部 layer)。

未建模 (MVP 范围外):
  - chunked send (流水线 prefill 完一段 send 一段, 重叠通信)
  - TP 重切 (prefill_tp != decode_tp 时 dst 侧需重排)
  - 多请求 batched send (NCCL coalesce)
"""
from __future__ import annotations


def kv_transfer_time(bytes_to_send: int, bandwidth_gbps: float, latency_us: float) -> float:
    """KV cache 跨节点传输 latency (秒).

    Args:
        bytes_to_send: KV cache 字节数 (req_kv_bytes())
        bandwidth_gbps: 带宽 GB/s (PD_CONNECTOR_PRESETS or 显式 override)
        latency_us: 起始 latency 微秒 (建链 / handshake / NIC queue)

    Returns:
        传输时间 (秒). bytes=0 时返 latency_us 部分 (空 send 仍有 handshake).
    """
    if bandwidth_gbps <= 0:
        return float("inf")
    latency_s = latency_us * 1e-6
    if bytes_to_send <= 0:
        return latency_s
    return latency_s + bytes_to_send / (bandwidth_gbps * 1e9)
