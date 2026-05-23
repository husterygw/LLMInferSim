"""Operator base layer — Protocol + RooflineSpec + legacy RooflineOperator/KVTransfer.

每个 op class 在 operators/{gemm,norm,elementwise,embedding,attention,collective,moe}.py
里独立; base.py 只提供 contract.

RooflineOperator / KVTransfer 是 ctx-pre 时代的通用 wrapper, 给 V3.2 indexer 这种
non-standard shape op 用. 新 op 类不该派生 RooflineOperator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from llm_infer_sim.core.operator_schema.signature import OperatorSignature


@dataclass(frozen=True)
class RooflineSpec:
    """Roofline/communication formula payload shared by all operator classes."""

    flops: int = 0
    load_weight: int = 0
    load_act: int = 0
    store_act: int = 0
    load_kv_cache: int = 0
    store_kv_cache: int = 0
    op_precision: str = ""
    comm_bytes: float = 0.0
    comm_type: str = ""
    op_category: str = ""

    @property
    def mem_bytes(self) -> int:
        return (
            self.load_weight
            + self.load_act
            + self.store_act
            + self.load_kv_cache
            + self.store_kv_cache
        )

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.mem_bytes if self.mem_bytes > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "flops": self.flops,
            "load_weight": self.load_weight,
            "load_act": self.load_act,
            "store_act": self.store_act,
            "load_kv_cache": self.load_kv_cache,
            "store_kv_cache": self.store_kv_cache,
            "op_precision": self.op_precision,
            "comm_bytes": self.comm_bytes,
            "comm_type": self.comm_type,
            "op_category": self.op_category,
        }


@runtime_checkable
class Operator(Protocol):
    """Semantic operator protocol consumed by cost backends."""

    name: str
    op_kind: str
    op_subtype: str
    phase: str
    layer_idx: int | None
    dtype: str
    dependencies: tuple[str, ...]
    tags: tuple[str, ...]

    @property
    def shape(self) -> dict[str, Any]: ...

    @property
    def parallel(self) -> dict[str, Any]: ...

    @property
    def runtime(self) -> dict[str, Any]: ...

    def signature(self) -> OperatorSignature: ...

    def roofline_spec(self) -> RooflineSpec: ...


# ============================================================================
# Legacy RooflineOperator / KVTransfer (ctx-pre 时代; V3.2 indexer 非标 shape op 用)
# ============================================================================

@dataclass(frozen=True)
class RooflineOperator:
    """Pre-computed roofline_spec_value + dict shape/parallel/runtime fields.

    新 op 类(GEMM/Norm/ElementWise/...) 不该派生这个; 它是给 V3.2 sparse_attn_indexer
    / indexer_q_fp8_quant 这种非标 shape op 用的兜底容器.
    """
    name: str
    op_kind: str
    op_subtype: str
    phase: str
    layer_idx: int | None
    dtype: str
    shape_fields: dict[str, Any]
    parallel_fields: dict[str, Any]
    runtime_fields: dict[str, Any]
    roofline_spec_value: RooflineSpec
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def shape(self) -> dict[str, Any]:
        return dict(self.shape_fields)

    @property
    def parallel(self) -> dict[str, Any]:
        return dict(self.parallel_fields)

    @property
    def runtime(self) -> dict[str, Any]:
        return dict(self.runtime_fields)

    def roofline_spec(self) -> RooflineSpec:
        return self.roofline_spec_value

    def signature(self) -> OperatorSignature:
        raise ValueError(
            f"{self.op_kind!r} is not in the OperatorDB signature contract"
        )


@dataclass(frozen=True)
class KVTransfer(RooflineOperator):
    """PD-disaggregation KV send/recv op (legacy RooflineOperator)."""
    pass
