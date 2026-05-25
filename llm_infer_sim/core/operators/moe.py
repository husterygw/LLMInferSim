"""MoE operators — moe_plan §3.3.

类层 (backend-neutral, AIC-aligned):
    MoE         routed expert compute kernel (was FusedMoE)
                op_kind=moe, op_subtype=routed_experts (default)
    MoEDispatch local reorder/pack/unpack/combine (无 collective)
                op_kind=moe_dispatch, op_subtype=pre_dispatch / post_dispatch
    MoERoutingProfile  routing distribution metadata for MoE class
    FusedMoE    deprecated alias to MoE (兼容旧 code path, 旧 op_subtype=fused_moe)

MoE GEMM helpers (moe_gate / shared_expert_up_gate / shared_expert_down) 不在此,
都是 op_kind=gemm 的 GEMM 实例, 由模型模板直接构造.

routing config (routing_distribution + power_law_alpha) 进 MoE signature key.
公式由 MoE.routed_experts(...) classmethod 内部计算.
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


# ---------------------------------------------------------------------------
# moe_plan Phase 4: 公式 helpers (拆自 MoE.routed_experts)
#
# 共享假设 (跟 AIC `MoE.get_weights` / `MoE.query` 字段对齐):
#   - 每 expert 含 gate + up + down = 3 个 linear weight, FMA = 2 flops/MAC
#     (Qwen3 / DeepSeek-V3 / Mixtral 都是 SiLU-gated 3-GEMM 结构;
#      vLLM fused_moe 可能 gate/up 融合成 1 个 GEMM, 但 FMA count 不变)
#   - moe_plan §4 Phase 4 handcheck 公式假设此处. 若 backend 用 gate_only/up_only
#     变体 (非 SiLU-gated), 由 backend-specific helper 在 calibration phase 调整.
# ---------------------------------------------------------------------------

def estimate_expert_token_distribution(
    *, tokens: int, top_k: int, num_experts: int, ep: int,
    skew: float = 0.0,
) -> dict[str, Any]:
    """估算单个 rank 上 expert/token 分布. 返回 dict 给 formula_breakdown.

    distinct_experts: 全局 expert 中实际被 hit 的数 (跟 coupon-collector + skew).
    tokens_per_device: 单 rank 跑的 token-expert 对数 (TP 时 = tokens, EP 时
                       = tokens × top_k / ep, 因 alltoall 后 token-expert 对分到
                       各 rank).
    """
    distinct = estimate_distinct_experts(
        tokens, top_k, num_experts, skew=skew,
    )
    if ep == 1:
        tokens_per_device = tokens
    else:
        tokens_per_device = tokens * top_k // ep
    return {
        "distinct_experts": distinct,
        "tokens_per_device": tokens_per_device,
    }


def estimate_expert_dim_per_device(
    *, expert_dim: int, tp: int, ep: int,
) -> int:
    """单 rank 持有的 expert intermediate dim.

    ep=1: TP 沿 intermediate 切, expert_dim_per_device = expert_dim // tp
    ep>1: EP 分 expert, 每 rank 持完整 expert (intermediate 不再 / tp,
          因 EP 跟 TP 不复用 sharding 维度)
    """
    return expert_dim // tp if ep == 1 else expert_dim


def estimate_grouped_gemm_flops(
    *, tokens: int, top_k: int, hidden: int,
    expert_dim_per_device: int, ep: int,
) -> int:
    """单 rank grouped GEMM 总 flops (SiLU-gated 3-GEMM: gate + up + down).

    公式: tokens × top_k × 3 (GEMM) × 2 (FMA) × hidden × expert_dim_per_device / ep.
    EP 时 token-expert 对均匀分 1/ep 给每 rank.
    """
    return tokens * top_k * 3 * 2 * hidden * expert_dim_per_device // ep


def estimate_grouped_gemm_weight_bytes(
    *, distinct_experts: float, hidden: int,
    expert_dim_per_device: int, w_byte: float, ep: int,
) -> int:
    """单 rank distinct expert 权重读取 (coupon-collector).

    公式: distinct × 3 (gate+up+down) × hidden × expert_dim_per_device × w_byte / ep.
    distinct 来自 estimate_distinct_experts (skew-aware).
    """
    return int(
        distinct_experts * 3 * hidden * expert_dim_per_device * w_byte / ep
    )


def estimate_moe_activation_bytes(
    *, tokens_per_device: int, hidden: int, a_byte: float,
) -> tuple[int, int]:
    """单 rank activation in/out bytes (load + store).

    grouped GEMM 输入 / 输出都是 tokens_per_device × hidden tensor.
    返回 (load, store).
    """
    bytes_each = int(tokens_per_device * hidden * a_byte)
    return bytes_each, bytes_each


# ============================================================================
# MoE op class (routed expert compute, backend-neutral)
# moe_plan §3.3: 默认 op_subtype="routed_experts"; kernel_source 表达 backend
# 实现 (vllm_fused_moe / vllm_marlin / trtllm / sglang). FusedMoE alias 保留.
# ============================================================================

@dataclass(frozen=True)
class MoE:
    """Backend-neutral routed expert compute. moe_plan §3.3.

    `op_subtype="routed_experts"` 是语义化命名; backend 信息靠
    `kernel_source` / `moe_backend` 表达 (e.g. vllm_fused_moe, trtllm,
    sglang_deepep). FusedMoE alias 保留作 deprecated 兼容入口.
    """
    name: str
    op_subtype: str    # 默认 "routed_experts"; 兼容传 "fused_moe"
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
    # moe_plan Phase 4: formula breakdown 数据透传到 CostTraceEntry.metadata
    formula_breakdown_value: dict[str, Any] = field(
        default_factory=dict, compare=False, hash=False, repr=False,
    )
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

    def formula_breakdown(self) -> dict[str, Any]:
        """moe_plan Phase 4: routed_experts 内部公式 breakdown.

        Keys (跟 plan §4 Phase 4 metadata 输出对齐):
            distinct_experts, expert_dim_per_device, tokens_per_device,
            expert_flops, expert_weight_read, routing_skew

        空 dict 兼容老 MoE op (没经过 routed_experts() classmethod 构造).
        """
        return dict(self.formula_breakdown_value)

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
        op_subtype: str = "routed_experts",
        kernel_source: str = "vllm_fused_moe",
        tags: tuple[str, ...] = (),
    ) -> "MoE":
        """Fused MoE routed experts (top_k semantics + coupon-collector distinct weight read).

        moe_plan Phase 4: 公式拆成 5 个 helpers, breakdown 进 formula_breakdown_value
        透传到 CostTraceEntry.metadata.

        切分:
          - ep=1: TP 沿 intermediate 切 expert_dim
          - ep>1: EP 分 expert (各 rank 持 1/ep 个 expert), intermediate 不再 / tp
        activation IO:
          - ep=1: tokens_per_device = tokens
          - ep>1: alltoall dispatch 后 tokens_per_device = tokens × top_k / ep
        """
        m = ctx.model
        tp = ctx.tp_size
        ep = ctx.ep_size
        top_k = m.num_activated_experts
        skew = routing.get_skew_for_layer(layer_idx)

        expert_dim_per_device = estimate_expert_dim_per_device(
            expert_dim=m.expert_dim, tp=tp, ep=ep,
        )
        dist = estimate_expert_token_distribution(
            tokens=tokens, top_k=top_k, num_experts=m.num_experts,
            ep=ep, skew=skew,
        )
        expert_flops = estimate_grouped_gemm_flops(
            tokens=tokens, top_k=top_k, hidden=m.hidden_dim,
            expert_dim_per_device=expert_dim_per_device, ep=ep,
        )
        expert_weight_read = estimate_grouped_gemm_weight_bytes(
            distinct_experts=dist["distinct_experts"],
            hidden=m.hidden_dim,
            expert_dim_per_device=expert_dim_per_device,
            w_byte=ctx.w_byte, ep=ep,
        )
        expert_act_in, expert_act_out = estimate_moe_activation_bytes(
            tokens_per_device=dist["tokens_per_device"],
            hidden=m.hidden_dim, a_byte=ctx.a_byte,
        )

        spec = RooflineSpec(
            flops=expert_flops,
            load_weight=expert_weight_read,
            load_act=expert_act_in,
            store_act=expert_act_out,
            op_precision="",
            op_category="matmul",
        )

        # moe_plan §4 Phase 4 metadata 输出 (透传到 CostTraceEntry.metadata)
        breakdown: dict[str, Any] = {
            "distinct_experts": float(dist["distinct_experts"]),
            "expert_dim_per_device": expert_dim_per_device,
            "tokens_per_device": dist["tokens_per_device"],
            "expert_flops": expert_flops,
            "expert_weight_read": expert_weight_read,
            "expert_act_load": expert_act_in,
            "expert_act_store": expert_act_out,
            "routing_skew": skew,
        }

        return cls(
            name=name, op_subtype=op_subtype,
            phase=phase, layer_idx=layer_idx,
            num_tokens=tokens,
            hidden=m.hidden_dim,
            moe_intermediate=m.expert_dim,
            topk=top_k,
            num_experts=m.num_experts,
            routing_distribution=routing.distribution,
            power_law_alpha=routing.power_law_alpha,
            ctx=ctx,
            kernel_source=kernel_source,
            tags=tags,
            roofline_spec_value=spec,
            formula_breakdown_value=breakdown,
        )


# ============================================================================
# FusedMoE: deprecated alias for MoE (moe_plan §3.3)
# 旧调用方 `FusedMoE(...)` / `FusedMoE.routed_experts(...)` 仍可工作.
# 跟 MoE 唯一差异: routed_experts() 默认 op_subtype="fused_moe" 而非
# "routed_experts", 用于 _legacy archive 数据兼容.
# ============================================================================

class FusedMoE(MoE):
    """[Deprecated] alias for MoE. 新代码请用 MoE.

    跟 MoE 同字段同行为, 唯一差异是 routed_experts() 默认 op_subtype="fused_moe"
    (保 collector raw record 兼容). 任何新 graph template / runtime path 都应该
    直接构造 MoE.
    """

    @classmethod
    def routed_experts(
        cls, *,
        layer_idx: int, tokens: int,
        ctx: OperatorContext, routing: MoERoutingProfile,
        phase: str = "decode",
        name: str = "routed_experts",
        op_subtype: str = "fused_moe",     # ← 保 legacy 默认
        kernel_source: str = "vllm_fused_moe",
        tags: tuple[str, ...] = (),
    ) -> "FusedMoE":
        return super().routed_experts(  # type: ignore[return-value]
            layer_idx=layer_idx, tokens=tokens, ctx=ctx, routing=routing,
            phase=phase, name=name, op_subtype=op_subtype,
            kernel_source=kernel_source, tags=tags,
        )


# ============================================================================
# MoEDispatch op class (local reorder / pack / unpack / combine, no collective)
# moe_plan §3.3 op#4 / op#8 + §3.4.1: backend communication 由并列 collective op
# 表达, MoEDispatch 仅承担 local kernel (align/sort/gather/permute).
# ============================================================================

@dataclass(frozen=True)
class MoEDispatch:
    """Local MoE dispatch/combine op. moe_plan §3.3 op#4 / op#8.

    pre_dispatch=True:  token permute / local reorder / pack 之前 communication
    pre_dispatch=False: expert output gather / unpack / combine 之后 communication

    实现边界 (§3.4.1): 禁止内部创建/调用 collective. 通信由并列 collective op
    (AllReduce / AllGather / ReduceScatter / AllToAll) 表达, 通过 metadata
    communication_peer 互链.
    """
    name: str
    pre_dispatch: bool
    phase: str
    layer_idx: int | None
    num_tokens: int
    hidden: int
    topk: int
    num_experts: int
    roofline_spec_value: RooflineSpec
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    dtype_override: str | None = None
    kernel_source: str = "vllm_fused_moe"     # internal MoE kernel includes dispatch
    communication_peer: str | None = None     # 关联 collective op 的 name (metadata 互链)
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def op_kind(self) -> str:
        return "moe_dispatch"

    @property
    def op_subtype(self) -> str:
        return "pre_dispatch" if self.pre_dispatch else "post_dispatch"

    @property
    def dtype(self) -> str:
        return self.dtype_override if self.dtype_override else self.ctx.dtype

    @property
    def shape(self) -> dict[str, Any]:
        return {
            "num_tokens": self.num_tokens,
            "hidden": self.hidden,
            "topk": self.topk,
            "num_experts": self.num_experts,
        }

    @property
    def parallel(self) -> dict[str, Any]:
        return {
            "tp": self.ctx.tp_size,
            "ep": self.ctx.ep_size,
        }

    @property
    def runtime(self) -> dict[str, Any]:
        md: dict[str, Any] = {
            "framework": self.ctx.framework,
            "framework_version": self.ctx.framework_version,
            "execution_mode": self.ctx.execution_mode,
            "kernel_source": self.kernel_source,
        }
        if self.communication_peer:
            md["communication_peer"] = self.communication_peer
        return md

    def roofline_spec(self) -> RooflineSpec:
        return self.roofline_spec_value

    @classmethod
    def local(
        cls, *,
        pre_dispatch: bool,
        layer_idx: int,
        tokens: int,
        ctx: OperatorContext,
        phase: str = "decode",
        name: str | None = None,
        communication_peer: str | None = None,
        tags: tuple[str, ...] = (),
    ) -> "MoEDispatch":
        """构造 local MoEDispatch op. roofline minimal placeholder (memory-only).

        load/store volume:
            pre_dispatch=True:  load tokens × hidden (raw activation),
                                store tokens × topk × hidden (fan-out by topk)
            pre_dispatch=False: load tokens × topk × hidden, store tokens × hidden

        moe_plan §5.A calibration 接 local_dispatch_overhead_us 后再调.
        Phase 3 用 memory-only spec 占位 (latency 量级跟 ElementWise 同).
        """
        m = ctx.model
        h = m.hidden_dim
        topk = m.num_activated_experts
        a = ctx.a_byte

        if pre_dispatch:
            load = tokens * h * a
            store = tokens * topk * h * a
        else:
            load = tokens * topk * h * a
            store = tokens * h * a

        spec = RooflineSpec(
            flops=0,
            load_act=int(load),
            store_act=int(store),
            op_category="memory",
        )

        default_name = (
            "moe_dispatch_pre" if pre_dispatch else "moe_dispatch_post"
        )
        return cls(
            name=name or default_name,
            pre_dispatch=pre_dispatch,
            phase=phase,
            layer_idx=layer_idx,
            num_tokens=tokens,
            hidden=h,
            topk=topk,
            num_experts=m.num_experts,
            ctx=ctx,
            communication_peer=communication_peer,
            tags=tags,
            roofline_spec_value=spec,
        )
