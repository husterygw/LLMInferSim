"""KVCacheConfig — KV cache 容量 (config_plan §4.3)。

block_size = paged-attention block 大小; num_gpu_blocks = profiling 后确定的总块数
(None = 尚未 profile)。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheConfig:
    block_size: int = 16
    num_gpu_blocks: int | None = None
