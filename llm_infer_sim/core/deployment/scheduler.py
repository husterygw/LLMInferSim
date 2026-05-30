"""SchedulerConfig — 调度容量 (config_plan §4.3)。

batch token / sequence 上限。None = 由 vLLM 默认 / 不限制。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
