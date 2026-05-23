"""Embedding operator — #158 ctx-based."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


@dataclass(frozen=True)
class Embedding:
    """Token embedding lookup: tokens × hidden (vocab × hidden weight)."""
    name: str
    phase: str
    layer_idx: int | None
    tokens: int
    vocab_size: int
    hidden: int
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    dtype_override: str | None = None
    kernel_source: str = "vllm_default"
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def op_kind(self) -> str:
        return "embedding"

    @property
    def op_subtype(self) -> str:
        return "embedding"

    @property
    def dtype(self) -> str:
        return self.dtype_override if self.dtype_override else self.ctx.dtype

    @property
    def shape(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "vocab_size": self.vocab_size,
            "hidden": self.hidden,
        }

    @property
    def parallel(self) -> dict[str, Any]:
        return {"tp": self.ctx.tp_size}

    @property
    def runtime(self) -> dict[str, Any]:
        return {
            "framework": self.ctx.framework,
            "framework_version": self.ctx.framework_version,
            "execution_mode": self.ctx.execution_mode,
            "kernel_source": self.kernel_source,
        }

    def roofline_spec(self) -> RooflineSpec:
        return RooflineSpec(
            op_category="embedding",
            flops=0,
            load_weight=int(self.vocab_size * self.hidden * self.ctx.w_byte),
            store_act=int(self.tokens * self.hidden * self.ctx.a_byte),
        )

    def signature(self) -> OperatorSignature:
        raise ValueError("embedding not in OperatorDB signature contract")
