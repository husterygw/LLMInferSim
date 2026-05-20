"""V3 §4.3 VirtualOp — 新架构 graph 层的中心对象.

VirtualOp 是:
    可查询 OperatorDB 的 op descriptor
  + roofline fallback feature (formula)
  + trace unit

字段语义 (V3 §4.3):
    op_kind / op_subtype / dtype / shape / parallel / runtime
        → OperatorDB query key
    formula
        → roofline / communication formula fallback
    name / layer_idx / phase / tags
        → trace 和报告

formula 首版字段 (IMPL_PLAN §1.4 Step 1.1):
    flops              — total compute (FLOPs)
    load_weight        — weight bytes loaded from HBM
    load_act           — input activation bytes loaded
    store_act          — output activation bytes stored
    load_kv_cache      — KV cache bytes loaded (attention only)
    store_kv_cache     — KV cache bytes stored (attention only)
    op_precision       — element bit width (8/16/...)
    comm_bytes         — collective message size (collective op only)
    comm_type          — allreduce / alltoall / ... (collective op only)
    op_category        — matmul / attention / norm / activation / communication / ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VirtualOp:
    name: str
    op_kind: str
    op_subtype: str
    phase: str
    layer_idx: int | None
    dtype: str
    shape: dict[str, Any]
    parallel: dict[str, Any]
    runtime: dict[str, Any]
    formula: dict[str, Any]
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
