"""MoEOpFactory — V3 §6.5 / IMPL_PLAN §4.

阶段 3a 范围 (Qwen3-30B-A3B 主线):
    moe_gate              — router GEMM (fp32 precision)
    routed_experts        — fused experts, top_k semantics + coupon-collector distinct
    shared_expert_*       — optional shared experts (V3 / Qwen3-Coder 用; A3B 没有)

未做 (后续阶段):
    V4 hash routing (num_hash_layers > 0) → moe_hash_lookup op   — 3c
    HC pre/post (hc_mult > 0)                                    — 3c
    expert FP4 (expert_fp4 + has_fp4_tc) precision override      — 3c

注意:
    通信 op (ep_alltoall_dispatch/combine, routed_expert_allreduce, shared_expert_allreduce)
    跟 mlp_norm / mlp_add 不由本 factory 生成. Qwen template 串完 orchestration.
"""
from __future__ import annotations

from llm_infer_sim.core.operators.factories.common import make_runtime
from llm_infer_sim.core.operators.ops import ElementwiseOp, FormulaOp, FusedMoeOp
from llm_infer_sim.core.operators.routing import (
    MoERoutingProfile,
    estimate_distinct_experts,
)
from llm_infer_sim.core.operators.specs import OperatorFormula
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig


class MoEOpFactory:
    def __init__(
        self,
        model: ModelConfig,
        deploy: DeployConfig,
        *,
        routing: MoERoutingProfile | None = None,
        w_byte: float = 2.0,
        a_byte: float = 2.0,
    ):
        self.model = model
        self.deploy = deploy
        self.routing = routing or MoERoutingProfile.balanced()
        self.w_byte = w_byte
        self.a_byte = a_byte

    # ---- router GEMM ----

    def moe_gate(self, layer_idx: int, tokens: int, phase: str) -> FormulaOp:
        """Router gate: m=tokens, n=num_experts, k=hidden. fp32 op_precision (路由稳定性)."""
        h = self.model.hidden_dim
        n_e = self.model.num_experts
        flops = 2 * tokens * h * n_e
        load_weight = int(h * n_e * self.w_byte)
        load_act = int(tokens * h * self.a_byte)
        store_act = int(tokens * n_e * self.a_byte)
        formula = OperatorFormula(
            flops=flops, load_weight=load_weight,
            load_act=load_act, store_act=store_act,
            op_precision="fp32", op_category="matmul",
        )
        return FormulaOp(
            name=f"moe_gate", op_kind="gemm", op_subtype="router",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"m": tokens, "n": n_e, "k": h},
            parallel_fields={"tp": self.deploy.tp_size},
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_moe_gate"),
            formula_value=formula,
        )

    def moe_hash_lookup(self, layer_idx: int, tokens: int, phase: str) -> ElementwiseOp:
        """V4 hash MoE: tid2eid 表查找, 替代 moe_gate router GEMM. FLOPs ≈ 0."""
        top_k = self.model.num_activated_experts
        return ElementwiseOp(
            name="moe_hash_lookup",
            op_kind="elementwise", op_subtype="moe_hash_lookup",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"tokens": tokens, "topk": top_k},
            parallel_fields={"tp": self.deploy.tp_size},
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_moe_hash_lookup"),
            formula_value=OperatorFormula(
                op_category="activation",
                flops=0,                                       # 查表, 无 GEMM
                load_act=int(tokens * 4),                      # token_id index (int32)
                store_act=int(tokens * top_k * 4),             # expert_ids (int32)
            ),
        )

    # ---- routed experts (fused_moe) ----

    def routed_experts(self, layer_idx: int, tokens: int, phase: str) -> FusedMoeOp:
        m = self.model
        tp = self.deploy.tp_size
        ep = self.deploy.ep_size
        h = m.hidden_dim
        top_k = m.num_activated_experts

        # 切分维度: ep=1 时 TP 切 expert_dim; ep>1 时 EP 分 expert 不切 dim
        expert_dim_per_device = m.expert_dim // tp if ep == 1 else m.expert_dim
        # 全部 tokens 的总 FFN FLOPs (per rank): tokens × top_k × 3 × 2 × h × dim / ep
        expert_flops = tokens * top_k * 3 * 2 * h * expert_dim_per_device // ep

        # weight: 单 step 实际只读到 distinct_experts 个 (coupon collector)
        skew = self.routing.get_skew_for_layer(layer_idx)
        distinct = estimate_distinct_experts(tokens, top_k, m.num_experts, skew=skew)
        expert_w_byte = self.w_byte
        expert_weight_read = int(
            distinct * 3 * h * expert_dim_per_device * expert_w_byte / ep
        )

        # activation IO:
        # - TP only (ep=1): 每 token 一份在每 rank, 各 token 各读一次
        # - EP (ep>1): AllToAll dispatch 后每 rank 收 tokens*top_k/ep 份
        tokens_per_device = tokens if ep == 1 else tokens * top_k // ep
        expert_act_in = int(tokens_per_device * h * self.a_byte)
        expert_act_out = int(tokens_per_device * h * self.a_byte)

        formula = OperatorFormula(
            flops=expert_flops,
            load_weight=expert_weight_read,
            load_act=expert_act_in,
            store_act=expert_act_out,
            op_precision="",   # follow global; fp4/fp8 override 留 3c
            op_category="matmul",
        )

        return FusedMoeOp(
            name="routed_experts", op_kind="moe", op_subtype="fused_moe",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={
                "num_tokens": tokens,
                "hidden": h,
                "moe_intermediate": m.expert_dim,
                "topk": top_k,
                "num_experts": m.num_experts,
                "routing_distribution": self.routing.distribution,
                "power_law_alpha": self.routing.power_law_alpha,
            },
            parallel_fields={"tp": tp, "ep": ep},
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_fused_moe"),
            formula_value=formula,
        )

    # ---- shared experts (optional) ----

    def shared_expert_up_gate(self, layer_idx: int, tokens: int, phase: str) -> FormulaOp:
        m = self.model
        tp = self.deploy.tp_size
        h = m.hidden_dim
        shared_dim = m.expert_dim * m.num_shared_experts
        shared_dim_per_device = shared_dim // tp
        flops = 2 * tokens * h * shared_dim_per_device * 2
        formula = OperatorFormula(
            flops=flops,
            load_weight=int(h * shared_dim_per_device * 2 * self.w_byte),
            load_act=int(tokens * h * self.a_byte),
            store_act=int(tokens * shared_dim_per_device * 2 * self.a_byte),
            op_precision="bf16", op_category="matmul",
        )
        return FormulaOp(
            name="shared_expert_up_gate", op_kind="gemm", op_subtype="shared_expert_up_gate",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"m": tokens, "n": 2 * shared_dim_per_device, "k": h},
            parallel_fields={"tp": tp},
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_row_parallel_linear"),
            formula_value=formula,
        )

    def shared_expert_act(self, layer_idx: int, tokens: int, phase: str) -> FormulaOp:
        m = self.model
        tp = self.deploy.tp_size
        shared_dim_per_device = m.expert_dim * m.num_shared_experts // tp
        formula = OperatorFormula(
            flops=5 * tokens * shared_dim_per_device,
            load_act=int(tokens * shared_dim_per_device * 2 * self.a_byte),
            store_act=int(tokens * shared_dim_per_device * self.a_byte),
            op_precision="bf16", op_category="activation",
        )
        return FormulaOp(
            name="shared_expert_act", op_kind="elementwise", op_subtype="silu_mul",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"tokens": tokens, "intermediate": shared_dim_per_device},
            parallel_fields={"tp": tp},
            runtime_fields=make_runtime(self.deploy),
            formula_value=formula,
        )

    def shared_expert_down(self, layer_idx: int, tokens: int, phase: str) -> FormulaOp:
        m = self.model
        tp = self.deploy.tp_size
        h = m.hidden_dim
        shared_dim_per_device = m.expert_dim * m.num_shared_experts // tp
        formula = OperatorFormula(
            flops=2 * tokens * shared_dim_per_device * h,
            load_weight=int(shared_dim_per_device * h * self.w_byte),
            load_act=int(tokens * shared_dim_per_device * self.a_byte),
            store_act=int(tokens * h * self.a_byte),
            op_precision="bf16", op_category="matmul",
        )
        return FormulaOp(
            name="shared_expert_down", op_kind="gemm", op_subtype="shared_expert_down",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"m": tokens, "n": h, "k": shared_dim_per_device},
            parallel_fields={"tp": tp},
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_row_parallel_linear"),
            formula_value=formula,
        )
