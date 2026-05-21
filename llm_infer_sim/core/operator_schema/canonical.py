"""Canonical helpers — dict → sorted tuple-of-tuple.

唯一的"如何把字典 normalize 成 hashable tuple"入口. 所有 canonicalizer 用它,
保证 GEMM / Attention / MoE / Collective 用同一套规则.
"""
from __future__ import annotations

from typing import Any


def to_canonical(fields: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """dict → sorted tuple-of-(key, value). 跳过 None value (保持 signature 最小)."""
    items = [(k, v) for k, v in fields.items() if v is not None]
    items.sort(key=lambda kv: kv[0])
    return tuple(items)


def project(d: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """从 dict 投影出指定 key, 缺失值为 None (供 to_canonical 跳过)."""
    return {k: d.get(k) for k in keys}
