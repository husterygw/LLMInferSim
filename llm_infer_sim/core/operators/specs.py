"""Operator layer contracts.

The runtime graph should be a list of semantic operators.  Cost backends decide
whether those operators are priced from measured data or from formulas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from llm_infer_sim.core.operator_schema.signature import OperatorSignature


@dataclass(frozen=True)
class OperatorFormula:
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

    def formula(self) -> OperatorFormula: ...
