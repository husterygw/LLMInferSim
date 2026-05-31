"""MoE operators — moe_plan §3.3.

类层 (backend-neutral, AIC-aligned):
    MoE         routed expert compute kernel; op_kind=moe,
                op_subtype=routed_experts (default; 旧 collector raw record 传 fused_moe)
    MoEDispatch dispatch 的跨卡通信 (AIC 对齐: MoEDispatch=通信, MoE=计算);
                op_kind=moe_dispatch, forward 按 tp/ep 解析 allreduce/本地
    MoERoutingProfile  routing distribution metadata for MoE class

build_routed_experts(ctx, routing, layer_idx, ...) / build_moe_dispatch(ctx, ...)
是模块级便捷构造器 (op 类不挂 classmethod 工厂, 跟 GEMM/Norm 一致), 封装 skew 折叠 +
从 ctx.model 派生字段, 给测试/脚本用. 模型图 (qwen3_moe/deepseek) 跟其它 op 一样
直接 MoE(...)/MoEDispatch(...) 构造. roofline_spec 从字段懒算.

MoE GEMM helpers (moe_gate / shared_expert_up_gate / shared_expert_down) 不在此,
都是 op_kind=gemm 的 GEMM 实例, 由模型模板直接构造.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from llm_infer_sim.core.step.runtime import OpRuntime, StepRuntime
from llm_infer_sim.core.operator_schema.moe import moe_operator_to_signature
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import OperatorBase, RooflineSpec
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
    """估算单个 rank 上 expert/token 分布 (routed_experts roofline 的中间量).

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


# ---------------------------------------------------------------------------
# Phase 4: shape-driven spec assembly (extracted from the named constructors so
# the spec can be recomputed from a step-resolved token count + static skew, and
# stays byte-identical to the legacy baked value — locked by tests).
# ---------------------------------------------------------------------------

def routed_experts_spec(
    *, tokens: int, skew: float, ctx: OperatorContext,
) -> RooflineSpec:
    """Recompute routed-experts RooflineSpec from tokens + static skew.

    skew is a STATIC per-op property (folded from MoERoutingProfile.layer_skews
    at construction by the model graph, one op per skew bucket) — it is NOT a
    step-runtime input. Only ``tokens`` varies per step.
    """
    m = ctx.model
    tp, ep = ctx.tp_size, ctx.ep_size
    top_k = m.num_activated_experts
    expert_dim_per_device = estimate_expert_dim_per_device(
        expert_dim=m.expert_dim, tp=tp, ep=ep,
    )
    dist = estimate_expert_token_distribution(
        tokens=tokens, top_k=top_k, num_experts=m.num_experts, ep=ep, skew=skew,
    )
    expert_flops = estimate_grouped_gemm_flops(
        tokens=tokens, top_k=top_k, hidden=m.hidden_dim,
        expert_dim_per_device=expert_dim_per_device, ep=ep,
    )
    expert_weight_read = estimate_grouped_gemm_weight_bytes(
        distinct_experts=dist["distinct_experts"], hidden=m.hidden_dim,
        expert_dim_per_device=expert_dim_per_device, w_byte=ctx.w_byte, ep=ep,
    )
    expert_act_in, expert_act_out = estimate_moe_activation_bytes(
        tokens_per_device=dist["tokens_per_device"], hidden=m.hidden_dim,
        a_byte=ctx.a_byte,
    )
    return RooflineSpec(
        flops=expert_flops,
        load_weight=expert_weight_read,
        load_act=expert_act_in,
        store_act=expert_act_out,
        op_precision="",
        op_category="matmul",
    )


def _moe_dispatch_spec(
    *, pre_dispatch: bool, tokens: int, hidden: int, topk: int, a_byte: float,
) -> RooflineSpec:
    if pre_dispatch:
        load, store = tokens * hidden * a_byte, tokens * topk * hidden * a_byte
    else:
        load, store = tokens * topk * hidden * a_byte, tokens * hidden * a_byte
    return RooflineSpec(
        flops=0, load_act=int(load), store_act=int(store), op_category="memory",
    )


def _moe_dispatch_comm(*, pre_dispatch: bool, tp: int, ep: int) -> tuple[str | None, int]:
    """MoEDispatch 的跨卡通信解析 (AIC 对齐: MoEDispatch=通信, MoE=计算).

    建模 vLLM 默认 MoE 路径 —— 本地 fused_experts(expert_map=...) + 单次
    tensor_model_parallel_all_reduce 聚合 partial sums:
      - pre_dispatch:  本地 permute/align, 无跨卡通信 → (None, 1)
      - post_dispatch: 跨 max(tp, ep) 卡 allreduce → ("allreduce", max(tp,ep))
    (TRT-LLM SM≥100 / SGLang-DeepEP 的 all2all dispatch/combine 是另一后端路径,
     当前不建模; 接入时此处按 backend 改返回 "alltoall".)
    """
    if pre_dispatch:
        return None, 1
    ws = max(tp, ep)
    return ("allreduce", ws) if ws > 1 else (None, 1)


# ============================================================================
# MoE op class (routed expert compute, backend-neutral)
# moe_plan §3.3: 默认 op_subtype="routed_experts"; kernel_source 表达 backend
# 实现 (vllm_fused_moe / vllm_marlin / trtllm / sglang).
# ============================================================================

@dataclass(frozen=True)
class MoE(OperatorBase):
    """Backend-neutral routed expert compute. moe_plan §3.3.

    `op_subtype="routed_experts"` 是语义化命名; backend 信息靠
    `kernel_source` / `moe_backend` 表达 (e.g. vllm_fused_moe, trtllm,
    sglang_deepep). 旧 collector raw record 用 op_subtype="fused_moe"。
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
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    # roofline_spec 默认从字段 (num_tokens + skew) 懒算 (像 GEMM 直接构造, 无需 baked
    # 值); 仅 op_runtime=None 的 fallback 路径用, build-once forward 永远给 op_runtime。
    roofline_spec_value: RooflineSpec | None = None
    dtype_override: str | None = None
    kernel_source: str = "vllm_fused_moe"
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    # Phase 4 static contract (op_plan §7/§440). count = MoE-layer multiplicity
    # within a routing-skew bucket; skew is the static (per-bucket) routing skew
    # folded in at construction (NOT a step input). Only num_tokens varies per step.
    count: int = 1
    skew: float = 0.0

    @property
    def op_kind(self) -> str:
        return "moe"

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

    # dtype / runtime 走 OperatorBase 默认; parallel 含 ep, 覆盖。

    @property
    def parallel(self) -> dict[str, Any]:
        return {"tp": self.ctx.tp_size, "ep": self.ctx.ep_size}

    def forward(self, step: StepRuntime) -> OpRuntime:
        return OpRuntime(
            phase=step.phase, op_subtype=None,  # signature subtype → fused_moe
            shape={
                "num_tokens": int(step.total_tokens), "hidden": self.hidden,
                "moe_intermediate": self.moe_intermediate, "topk": self.topk,
                "num_experts": self.num_experts,
                "routing_distribution": self.routing_distribution,
                "power_law_alpha": self.power_law_alpha,
            },
            parallel=dict(self.parallel), runtime=dict(self.runtime),
        )

    def roofline_spec(self, op_runtime: OpRuntime | None = None) -> RooflineSpec:
        if op_runtime is None:
            if self.roofline_spec_value is not None:
                return self.roofline_spec_value
            return routed_experts_spec(tokens=self.num_tokens, skew=self.skew, ctx=self.ctx)
        return routed_experts_spec(
            tokens=int(op_runtime.shape["num_tokens"]), skew=self.skew, ctx=self.ctx,
        )

    def signature(self, op_runtime: OpRuntime | None = None) -> OperatorSignature:
        # 单一契约入口: 委托给 canonicalizer, 它负责 subtype 归一到 fused_moe +
        # per-kernel parallel (moe_tp/moe_ep)。不要在此重复构造, 否则两处会漂移
        # (历史 bug: inline 构造用了 self.op_subtype="routed_experts" + 全局 tp)。
        return moe_operator_to_signature(self, op_runtime)


# ============================================================================
# MoEDispatch op class (AIC 对齐: MoEDispatch=通信, MoE=计算).
# moe_plan §3.3 op#4 / op#8. forward(step) 按 tp/ep 解析 dispatch 的跨卡通信:
#   pre_dispatch  → 本地 permute/align (vLLM 无跨卡通信) → op_subtype=None
#   post_dispatch → tensor_model_parallel_all_reduce(world=max(tp,ep)) 或本地
# 成本走 RooflineBackend._estimate_collective (与 Collective 同公式).
# ============================================================================

@dataclass(frozen=True)
class MoEDispatch(OperatorBase):
    """MoE dispatch/combine op — 承载 dispatch 的跨卡通信. moe_plan §3.3 op#4 / op#8.

    forward(step) 由 (pre_dispatch, tp, ep) 解析出 op_subtype:
      - "allreduce": post-dispatch 且 max(tp,ep)>1 (vLLM 默认 partial-sum 聚合)
      - None:        本地 (pre-dispatch, 或 max(tp,ep)==1) → message_bytes=0, comm 0
    (TRT-LLM SM≥100 / SGLang-DeepEP 的 all2all dispatch/combine 是另一后端, 接入时
     在 _moe_dispatch_comm 按 backend 返回 "alltoall".)
    """
    name: str
    pre_dispatch: bool
    phase: str
    layer_idx: int | None
    num_tokens: int
    hidden: int
    topk: int
    num_experts: int
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    # 本地 spec 默认从字段懒算 (像 GEMM 直接构造, 无需 baked 值); 仅 op_runtime=None
    # 的 legacy 路径用. 通信由 forward 解析 + _estimate_collective 计算.
    roofline_spec_value: RooflineSpec | None = None
    dtype_override: str | None = None
    kernel_source: str = "vllm_fused_moe"     # internal MoE kernel includes dispatch
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    count: int = 1                            # Phase 4 static contract (MoE-layer multiplicity)

    @property
    def op_kind(self) -> str:
        return "moe_dispatch"

    @property
    def op_subtype(self) -> str:
        return "pre_dispatch" if self.pre_dispatch else "post_dispatch"

    @property
    def shape(self) -> dict[str, Any]:
        return {
            "num_tokens": self.num_tokens,
            "hidden": self.hidden,
            "topk": self.topk,
            "num_experts": self.num_experts,
        }

    # dtype / runtime 走 OperatorBase 默认; parallel 含 ep, 覆盖。

    @property
    def parallel(self) -> dict[str, Any]:
        return {
            "tp": self.ctx.tp_size,
            "ep": self.ctx.ep_size,
        }

    def forward(self, step: StepRuntime) -> OpRuntime:
        """Resolve the dispatch communication from tp/ep (AIC 对齐: dispatch=通信).

        op_subtype = 解析出的 collective ("allreduce") 或 None (本地, 无跨卡通信);
        cost 走 RooflineBackend._estimate_collective. 始终返回非 None (op 恒在图里,
        承载 local_dispatch_overhead calibration; 无通信时 message_bytes=0 → comm 0)."""
        tp, ep = self.ctx.tp_size, self.ctx.ep_size
        comm_type, world_size = _moe_dispatch_comm(
            pre_dispatch=self.pre_dispatch, tp=tp, ep=ep,
        )
        mb = int(step.total_tokens * self.hidden * self.ctx.a_byte) if comm_type else 0
        return OpRuntime(
            phase=step.phase, op_subtype=comm_type,
            shape={"message_bytes": mb, "num_tokens": int(step.total_tokens),
                   "hidden": self.hidden, "topk": self.topk,
                   "num_experts": self.num_experts},
            parallel={"world_size": world_size, "tp": tp, "ep": ep},
            runtime=dict(self.runtime),
        )

    def roofline_spec(self, op_runtime: OpRuntime | None = None) -> RooflineSpec:
        # comm-resolved (op_runtime.op_subtype set) → communication spec; else
        # the local permute/align memory spec (legacy / DB fallback).
        if op_runtime is not None and op_runtime.op_subtype:
            return RooflineSpec(
                comm_bytes=float(op_runtime.shape.get("message_bytes", 0)),
                comm_type=op_runtime.op_subtype, op_category="communication",
            )
        tokens = self.num_tokens if op_runtime is None else int(op_runtime.shape["num_tokens"])
        if op_runtime is None and self.roofline_spec_value is not None:
            return self.roofline_spec_value
        return _moe_dispatch_spec(
            pre_dispatch=self.pre_dispatch, tokens=tokens,
            hidden=self.hidden, topk=self.topk, a_byte=self.ctx.a_byte,
        )


# ============================================================================
# 模块级 build 函数 (替代旧的 MoE.routed_experts / MoEDispatch.local classmethod).
# op 类本身不挂工厂方法 (跟 GEMM/Norm 一致); 这两个 op 需要从 ctx.model + routing
# 派生字段 (skew 折叠 / hidden / expert_dim / topk...), 故由 build_* 封装一处.
# tokens=0 是 build-once 占位 (forward 从 step 重算); per-step / 测试传真实 tokens.
# spec / breakdown 懒算 (无需 baked 值).
# ============================================================================

def build_routed_experts(
    ctx: OperatorContext, routing: MoERoutingProfile, layer_idx: int, *,
    tokens: int = 0, count: int = 1, phase: str = "decode",
    name: str = "routed_experts", op_subtype: str = "routed_experts",
    kernel_source: str = "vllm_fused_moe", tags: tuple[str, ...] = (),
) -> MoE:
    """Routed-experts MoE op. skew = routing.get_skew_for_layer(layer_idx) 折成静态
    per-op 参数 (op_plan §440, 不是 step 输入); 其余字段从 ctx.model 派生."""
    m = ctx.model
    return MoE(
        name=name, op_subtype=op_subtype, phase=phase, layer_idx=layer_idx,
        num_tokens=tokens, hidden=m.hidden_dim, moe_intermediate=m.expert_dim,
        topk=m.num_activated_experts, num_experts=m.num_experts,
        routing_distribution=routing.distribution,
        power_law_alpha=routing.power_law_alpha,
        skew=routing.get_skew_for_layer(layer_idx),
        ctx=ctx, kernel_source=kernel_source, tags=tags, count=count,
    )


def build_moe_dispatch(
    ctx: OperatorContext, *, pre_dispatch: bool, layer_idx: int = 0,
    tokens: int = 0, count: int = 1, phase: str = "decode",
    name: str | None = None, tags: tuple[str, ...] = (),
) -> MoEDispatch:
    """MoE dispatch op. forward() 按 tp/ep 解析跨卡通信 (allreduce / 本地);
    字段从 ctx.model 派生."""
    m = ctx.model
    return MoEDispatch(
        name=name or ("moe_dispatch_pre" if pre_dispatch else "moe_dispatch_post"),
        pre_dispatch=pre_dispatch, phase=phase, layer_idx=layer_idx,
        num_tokens=tokens, hidden=m.hidden_dim, topk=m.num_activated_experts,
        num_experts=m.num_experts, ctx=ctx, tags=tags, count=count,
    )
