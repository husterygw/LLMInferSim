"""MoE operator — FusedMoE class + MoERoutingProfile + estimate_distinct_experts.

FusedMoE 是 routed experts kernel. shape 含 routing config (routing_distribution +
power_law_alpha) 进 OperatorDB signature. 公式由 FusedMoE.routed_experts(...)
classmethod 内部计算.

MoE GEMM helpers (moe_gate / shared_expert_up_gate / shared_expert_down) 不在此,
都是 op_kind=gemm 的 GEMM 实例, 由模型模板直接构造 (传 op_precision_override 处理
fp32/bf16 边界).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


# ============================================================================
# MoE routing profile
# ============================================================================

@dataclass(frozen=True)
class MoERoutingProfile:
    """Routing distribution + per-layer skew 估算 distinct experts.

    distribution: "balanced" (uniform) / "power_law" (token-imbalanced).
    power_law_alpha: power-law 偏度 (越大越偏向 top expert; balanced 忽略).
    skew: global skew override (默认 0 = balanced; 1 = 退化到 top_k worst-case).
    layer_skews: 可选 per-layer skew override; 越界回 global skew.
    """
    distribution: Literal["balanced", "power_law"] = "balanced"
    power_law_alpha: float = 0.0
    skew: float = 0.0
    layer_skews: tuple[float, ...] = ()

    @classmethod
    def balanced(cls) -> "MoERoutingProfile":
        return cls(distribution="balanced", power_law_alpha=0.0, skew=0.0)

    @classmethod
    def power_law(cls, alpha: float, skew: float = 0.0) -> "MoERoutingProfile":
        return cls(distribution="power_law", power_law_alpha=alpha, skew=skew)

    def get_skew_for_layer(self, layer_idx: int) -> float:
        if self.layer_skews and layer_idx < len(self.layer_skews):
            return self.layer_skews[layer_idx]
        return self.skew


def estimate_distinct_experts(
    tokens: int, top_k: int, num_experts: int, *, skew: float = 0.0,
) -> float:
    """Coupon-collector 估算: T 个 token, 每 token 选 top_k expert, 共 num_experts.

    实际命中的 distinct expert 数 ≈ N × (1 - (1 - k/N)^T).

    skew clip 到 [0,1]:
      - skew=0 (balanced): coupon-collector
      - skew=1 (worst): 退化到 top_k (所有 token 命中同 top_k 个 expert)
    """
    total_draws = tokens * top_k
    if num_experts <= 0 or total_draws <= 0:
        return 0.0
    skew = max(0.0, min(1.0, skew))
    balanced = num_experts * (1.0 - (1.0 - top_k / num_experts) ** tokens)
    if skew >= 1.0:
        return float(top_k)
    return (1.0 - skew) * balanced + skew * float(top_k)


# ============================================================================
# FusedMoE op class
# ============================================================================

@dataclass(frozen=True)
class FusedMoE:
    """Fused MoE routed experts kernel."""
    name: str
    op_subtype: str    # 一般 "fused_moe"
    phase: str
    layer_idx: int | None
    num_tokens: int
    hidden: int
    moe_intermediate: int
    topk: int
    num_experts: int
    routing_distribution: str
    power_law_alpha: float
    roofline_spec_value: RooflineSpec
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    dtype_override: str | None = None
    kernel_source: str = "vllm_fused_moe"
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def op_kind(self) -> str:
        return "moe"

    @property
    def dtype(self) -> str:
        return self.dtype_override if self.dtype_override else self.ctx.dtype

    @property
    def shape(self) -> dict[str, Any]:
        return {
            "num_tokens": self.num_tokens,
            "hidden": self.hidden,
            "moe_intermediate": self.moe_intermediate,
            "topk": self.topk,
            "num_experts": self.num_experts,
            "routing_distribution": self.routing_distribution,
            "power_law_alpha": self.power_law_alpha,
        }

    @property
    def parallel(self) -> dict[str, Any]:
        return {"tp": self.ctx.tp_size, "ep": self.ctx.ep_size}

    @property
    def runtime(self) -> dict[str, Any]:
        return {
            "framework": self.ctx.framework,
            "framework_version": self.ctx.framework_version,
            "execution_mode": self.ctx.execution_mode,
            "kernel_source": self.kernel_source,
        }

    def roofline_spec(self) -> RooflineSpec:
        return self.roofline_spec_value

    def signature(self) -> OperatorSignature:
        return OperatorSignature(
            op_kind="moe",
            op_subtype=self.op_subtype,
            dtype=self.dtype,
            shape=to_canonical(project(self.shape, (
                "num_tokens", "hidden", "moe_intermediate", "topk",
                "num_experts", "routing_distribution", "power_law_alpha",
            ))),
            parallel=to_canonical(project(self.parallel, ("tp", "ep"))),
            runtime=to_canonical(project(self.runtime, (
                "framework", "framework_version", "execution_mode", "kernel_source",
            ))),
        )

    # ---- named constructor ----

    @classmethod
    def routed_experts(
        cls, *,
        layer_idx: int, tokens: int,
        ctx: OperatorContext, routing: MoERoutingProfile,
        phase: str = "decode",
        name: str = "routed_experts",
        op_subtype: str = "fused_moe",
        kernel_source: str = "vllm_fused_moe",
        tags: tuple[str, ...] = (),
    ) -> "FusedMoE":
        """Fused MoE routed experts (top_k semantics + coupon-collector distinct weight read).

        切分:
          - ep=1: TP 切 expert_dim
          - ep>1: EP 分 expert (各 rank 持 1/ep 个 expert), expert_dim 不再 / tp
        activation IO:
          - ep=1: 每 token 一份在 rank 上, tokens/rank = tokens
          - ep>1: alltoall dispatch 后 tokens × top_k / ep / rank
        """
        m = ctx.model
        tp = ctx.tp_size
        ep = ctx.ep_size
        h = m.hidden_dim
        top_k = m.num_activated_experts

        expert_dim_per_device = m.expert_dim // tp if ep == 1 else m.expert_dim
        expert_flops = tokens * top_k * 3 * 2 * h * expert_dim_per_device // ep

        skew = routing.get_skew_for_layer(layer_idx)
        distinct = estimate_distinct_experts(tokens, top_k, m.num_experts, skew=skew)
        expert_weight_read = int(
            distinct * 3 * h * expert_dim_per_device * ctx.w_byte / ep
        )

        tokens_per_device = tokens if ep == 1 else tokens * top_k // ep
        expert_act_in = int(tokens_per_device * h * ctx.a_byte)
        expert_act_out = int(tokens_per_device * h * ctx.a_byte)

        spec = RooflineSpec(
            flops=expert_flops,
            load_weight=expert_weight_read,
            load_act=expert_act_in,
            store_act=expert_act_out,
            op_precision="",
            op_category="matmul",
        )
        return cls(
            name=name, op_subtype=op_subtype,
            phase=phase, layer_idx=layer_idx,
            num_tokens=tokens,
            hidden=m.hidden_dim,
            moe_intermediate=m.expert_dim,
            topk=m.num_activated_experts,
            num_experts=m.num_experts,
            routing_distribution=routing.distribution,
            power_law_alpha=routing.power_law_alpha,
            ctx=ctx,
            kernel_source=kernel_source,
            tags=tags,
            roofline_spec_value=spec,
        )
