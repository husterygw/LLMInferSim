"""Embedding operator — #158 ctx-based."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from llm_infer_sim.core.step.runtime import OpRuntime, StepRuntime
from llm_infer_sim.core.operators.base import OperatorBase, RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


@dataclass(frozen=True)
class Embedding(OperatorBase):
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
    # Phase 3 static contract (op_plan §6/§7). embedding/lm_head count=1.
    count: int = 1
    tokens_fn: Callable[[StepRuntime], int] | None = field(
        default=None, compare=False, hash=False, repr=False,
    )

    @property
    def op_kind(self) -> str:
        return "embedding"

    @property
    def op_subtype(self) -> str:
        return "embedding"

    @property
    def shape(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "vocab_size": self.vocab_size,
            "hidden": self.hidden,
        }

    # dtype / parallel / runtime / signature(raise) 走 OperatorBase 默认。

    def forward(self, step: StepRuntime) -> OpRuntime:
        tokens = self.tokens_fn(step) if self.tokens_fn is not None else self.tokens
        return OpRuntime(
            phase=step.phase, op_subtype=None,
            shape={"tokens": int(tokens), "vocab_size": self.vocab_size, "hidden": self.hidden},
            parallel=dict(self.parallel), runtime=dict(self.runtime),
        )

    def roofline_spec(self, op_runtime: OpRuntime | None = None) -> RooflineSpec:
        tokens = int(op_runtime.shape["tokens"]) if op_runtime is not None else self.tokens
        return RooflineSpec(
            op_category="embedding",
            flops=0,
            load_weight=int(self.vocab_size * self.hidden * self.ctx.w_byte),
            store_act=int(tokens * self.hidden * self.ctx.a_byte),
        )
