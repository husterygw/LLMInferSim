"""ParallelismConfig — 部署并行度 (config_plan §4.3).

只描述并行切分, 不含 scheduler 容量 / KV cache / runtime execution mode。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelismConfig:
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
    moe_tp: int = 1
    moe_ep: int = 1
