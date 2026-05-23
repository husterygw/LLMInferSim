"""Norm operator (RMSNorm / LayerNorm) — #158 ctx-based."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


@dataclass(frozen=True)
class Norm:
    """RMSNorm / LayerNorm. flops = tokens × hidden × 4 (RMSNorm-ish)."""
    name: str
    op_subtype: str
    phase: str
    layer_idx: int | None
    tokens: int
    hidden: int
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    dtype_override: str | None = None
    kernel_source: str = "vllm_default"
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def op_kind(self) -> str:
        return "norm"

    @property
    def dtype(self) -> str:
        return self.dtype_override if self.dtype_override else self.ctx.dtype

    @property
    def shape(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "hidden": self.hidden}

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
        elements = self.tokens * self.hidden
        a_byte = self.ctx.a_byte
        return RooflineSpec(
            op_category="norm",
            flops=elements * 4,
            load_act=int(elements * a_byte),
            store_act=int(elements * a_byte),
        )

    def signature(self) -> OperatorSignature:
        raise ValueError("norm not in OperatorDB signature contract")
