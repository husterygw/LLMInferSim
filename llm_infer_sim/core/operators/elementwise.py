"""ElementWise operator (rope / attn_add / mlp_add / mlp_act / silu_mul) — #158."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


@dataclass(frozen=True)
class ElementWise:
    """Elementwise ops (rope / attn_add / mlp_add / mlp_act / silu_mul).

    roofline_spec() 按 op_subtype 分发. 后续每种 subtype 可以拆成独立 class, 当前先合并.
    """
    name: str
    op_subtype: str
    phase: str
    layer_idx: int | None
    tokens: int
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    # subtype-specific shape inputs (only ones relevant for the subtype)
    hidden: int = 0           # for attn_add / mlp_add / rope
    intermediate: int = 0     # for mlp_act / silu_mul
    num_heads: int = 0        # for rope
    head_dim: int = 0         # for rope
    dtype_override: str | None = None
    kernel_source: str = "vllm_default"
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def op_kind(self) -> str:
        return "elementwise"

    @property
    def dtype(self) -> str:
        return self.dtype_override if self.dtype_override else self.ctx.dtype

    @property
    def shape(self) -> dict[str, Any]:
        out = {"tokens": self.tokens}
        if self.hidden:
            out["hidden"] = self.hidden
        if self.intermediate:
            out["intermediate"] = self.intermediate
        if self.num_heads:
            out["num_heads"] = self.num_heads
        if self.head_dim:
            out["head_dim"] = self.head_dim
        return out

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
        a_byte = self.ctx.a_byte
        sub = self.op_subtype
        if sub in ("attn_add", "mlp_add"):
            elements = self.tokens * self.hidden
            return RooflineSpec(
                op_category="activation",
                flops=elements,
                load_act=int(elements * a_byte),
                store_act=int(elements * a_byte),
            )
        if sub in ("mlp_act", "silu_mul"):
            # SiLU(gate) * up: 5 flops/elem, read 2 (gate+up), write 1.
            elements = self.tokens * self.intermediate
            return RooflineSpec(
                op_category="activation",
                flops=elements * 5,
                load_act=int(elements * a_byte * 2),
                store_act=int(elements * a_byte),
            )
        if sub == "rope":
            # apply rope to Q + K: caller passes num_heads = n_q_per_tp + n_kv_per_tp.
            elements = self.tokens * self.num_heads * self.head_dim
            return RooflineSpec(
                op_category="activation",
                flops=elements * 3,
                load_act=int(elements * a_byte),
                store_act=int(elements * a_byte),
            )
        if sub == "topk":
            # moe_plan §3.3 op#3: softmax + topk over (tokens, num_experts).
            # caller 把 num_experts 传 self.intermediate.
            # softmax 占主导 (~5 flops/elem); topk K-select 量级小, placeholder 不细拆.
            elements = self.tokens * self.intermediate
            return RooflineSpec(
                op_category="activation",
                flops=elements * 5,
                load_act=int(elements * a_byte),
                store_act=int(elements * a_byte),
            )
        raise ValueError(f"unsupported elementwise subtype: {sub!r}")

    def signature(self) -> OperatorSignature:
        raise ValueError("elementwise not in OperatorDB signature contract")
