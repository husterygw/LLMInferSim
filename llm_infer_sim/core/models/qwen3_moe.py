"""Qwen3-MoE model flat builder.

The model class builds a static op list in ``__init__`` and ``forward`` only
attaches step runtime. MoE compute and local dispatch are represented by the
shared semantic operator classes ``MoE`` and ``MoEDispatch``.
"""
from __future__ import annotations

import os

from llm_infer_sim.core.graph.runtime import StepRuntime
from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.models.layer_partition import partition_ffn_layers
from llm_infer_sim.core.operators import (
    AllReduce,
    Attention,
    ElementWise,
    Embedding,
    GEMM,
    MoE,
    MoEDispatch,
    MoERoutingProfile,
    Norm,
)
from llm_infer_sim.core.operators.base import Operator
from llm_infer_sim.core.operators.context import OperatorContext
from llm_infer_sim.core.models.registry import register_model
from llm_infer_sim.core.models.config import ModelProfile

_STATIC_PHASE = ""
_TP_COLLECTIVE_KERNEL_SOURCE = "torch_dist_nccl"


def _num_tokens(step: StepRuntime) -> int:
    return step.total_tokens


def _lm_head_m(step: StepRuntime) -> int:
    if step.phase == "prefill":
        return max(step.num_prefill_requests, 1)
    return step.num_decode_requests


def _message_bytes_hidden(ctx: OperatorContext):
    return lambda step: int(step.total_tokens * ctx.model.hidden_dim * ctx.a_byte)


def _topology_hint() -> str:
    hint = os.environ.get("LLM_INFER_SIM_NUMA_HINT", "concentrated")
    return hint if hint in ("concentrated", "balanced") else "concentrated"


def _shared_dim_per_tp(ctx: OperatorContext) -> int:
    return ctx.model.expert_dim * ctx.model.num_shared_experts // ctx.tp_size


class Qwen3MoeModel:
    """Qwen3-MoE op graph.

    The attention structure is uniform across layers. FFN layers are bucketed by
    dense/MoE kind so configs with Qwen mlp-only leading layers can still be
    represented without constructing per-step operators.
    """

    def __init__(
        self,
        model_config: ModelProfile,
        ctx: OperatorContext,
    ) -> None:
        if model_config.num_experts <= 0:
            raise ValueError(
                "Qwen3MoeModel requires num_experts > 0; use Qwen3Model"
            )
        self.model_config = model_config
        self.ctx = ctx
        self.ops: tuple[Operator, ...] = tuple(self._build_ops())

    def _build_ops(self) -> list[Operator]:
        ops: list[Operator] = []
        ops.extend(self._embedding_ops())
        ops.extend(self._attention_ops(count=self.model_config.num_layers))
        for ffn_kind, layer_indices in partition_ffn_layers(self.model_config):
            if not layer_indices:
                continue
            rep_layer_idx = layer_indices[0]
            count = len(layer_indices)
            if ffn_kind == "dense":
                ops.extend(self._dense_ffn_ops(count=count))
            elif ffn_kind == "moe":
                ops.extend(
                    self._moe_ffn_ops(
                        count=count,
                        rep_layer_idx=rep_layer_idx,
                    )
                )
            else:
                raise ValueError(f"Unsupported Qwen3 FFN kind: {ffn_kind!r}")
        ops.extend(self._lm_head_ops())
        return ops

    def _embedding_ops(self) -> list[Operator]:
        model_config = self.model_config
        return [
            Embedding(
                name="embedding",
                phase=_STATIC_PHASE,
                layer_idx=None,
                tokens=0,
                vocab_size=model_config.vocab_size,
                hidden=model_config.hidden_dim,
                ctx=self.ctx,
                tokens_fn=_num_tokens,
            )
        ]

    def _attention_ops(self, *, count: int) -> list[Operator]:
        model_config = self.model_config
        ctx = self.ctx
        tp = ctx.tp_size
        n_q = model_config.num_heads // tp
        n_kv = model_config.num_kv_heads // tp
        qkv_out = (n_q + 2 * n_kv) * model_config.head_dim
        q_out = n_q * model_config.head_dim

        ops: list[Operator] = [
            Norm(
                name="attn_norm",
                op_subtype="attn_norm",
                phase=_STATIC_PHASE,
                layer_idx=0,
                tokens=0,
                hidden=model_config.hidden_dim,
                ctx=ctx,
                count=count,
                tokens_fn=_num_tokens,
            ),
            GEMM(
                name="qkv_proj",
                op_subtype="qkv_proj",
                phase=_STATIC_PHASE,
                layer_idx=0,
                m=0,
                n=qkv_out,
                k=model_config.hidden_dim,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
                count=count,
                m_fn=_num_tokens,
            ),
            ElementWise(
                name="rope",
                op_subtype="rope",
                phase=_STATIC_PHASE,
                layer_idx=0,
                tokens=0,
                num_heads=n_q + n_kv,
                head_dim=model_config.head_dim,
                ctx=ctx,
                count=count,
                tokens_fn=_num_tokens,
            ),
            # 单 attention op: forward(step) 解析 prefill/decode/mixed
            Attention.flash(
                name="attention",
                layer_idx=0,
                n_q=n_q,
                n_kv=n_kv,
                head_dim=model_config.head_dim,
                ctx=ctx,
                phase=_STATIC_PHASE,
                count=count,
            ),
            GEMM(
                name="o_proj",
                op_subtype="o_proj",
                phase=_STATIC_PHASE,
                layer_idx=0,
                m=0,
                n=model_config.hidden_dim,
                k=q_out,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
                count=count,
                m_fn=_num_tokens,
            ),
        ]
        # build-once: always emit the TP allreduce; world_size=tp gates it at
        # runtime (tp=1 → Collective.forward()->None → router skips, no cost).
        ops.append(
            AllReduce(
                name="tp_o_proj_allreduce",
                phase=_STATIC_PHASE,
                layer_idx=0,
                message_bytes=0,
                world_size=tp,
                ctx=ctx,
                kernel_source=_TP_COLLECTIVE_KERNEL_SOURCE,
                topology=_topology_hint(),
                count=count,
                message_bytes_fn=_message_bytes_hidden(ctx),
            )
        )
        ops.append(
            ElementWise(
                name="attn_add",
                op_subtype="attn_add",
                phase=_STATIC_PHASE,
                layer_idx=0,
                tokens=0,
                hidden=model_config.hidden_dim,
                ctx=ctx,
                count=count,
                tokens_fn=_num_tokens,
            )
        )
        return ops

    def _dense_ffn_ops(self, *, count: int) -> list[Operator]:
        model_config = self.model_config
        ctx = self.ctx
        tp = ctx.tp_size
        ffn = model_config.ffn_dim // tp

        ops: list[Operator] = [
            Norm(
                name="mlp_norm",
                op_subtype="mlp_norm",
                phase=_STATIC_PHASE,
                layer_idx=0,
                tokens=0,
                hidden=model_config.hidden_dim,
                ctx=ctx,
                count=count,
                tokens_fn=_num_tokens,
            ),
            GEMM(
                name="gate_up_proj",
                op_subtype="gate_up_proj",
                phase=_STATIC_PHASE,
                layer_idx=0,
                m=0,
                n=2 * ffn,
                k=model_config.hidden_dim,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
                count=count,
                m_fn=_num_tokens,
            ),
            ElementWise(
                name="mlp_act",
                op_subtype="mlp_act",
                phase=_STATIC_PHASE,
                layer_idx=0,
                tokens=0,
                intermediate=ffn,
                ctx=ctx,
                count=count,
                tokens_fn=_num_tokens,
            ),
            GEMM(
                name="down_proj",
                op_subtype="down_proj",
                phase=_STATIC_PHASE,
                layer_idx=0,
                m=0,
                n=model_config.hidden_dim,
                k=ffn,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
                count=count,
                m_fn=_num_tokens,
            ),
        ]
        ops.append(
            AllReduce(
                name="tp_down_proj_allreduce",
                phase=_STATIC_PHASE,
                layer_idx=0,
                message_bytes=0,
                world_size=tp,
                ctx=ctx,
                kernel_source=_TP_COLLECTIVE_KERNEL_SOURCE,
                topology=_topology_hint(),
                count=count,
                message_bytes_fn=_message_bytes_hidden(ctx),
            )
        )
        ops.append(
            ElementWise(
                name="mlp_add",
                op_subtype="mlp_add",
                phase=_STATIC_PHASE,
                layer_idx=0,
                tokens=0,
                hidden=model_config.hidden_dim,
                ctx=ctx,
                count=count,
                tokens_fn=_num_tokens,
            )
        )
        return ops

    def _moe_ffn_ops(
        self,
        *,
        count: int,
        rep_layer_idx: int,
    ) -> list[Operator]:
        model_config = self.model_config
        ctx = self.ctx
        # routing 随 ctx 传入 (部署级成本假设); None → balanced().
        routing = ctx.routing or MoERoutingProfile.balanced()

        ops: list[Operator] = [
            Norm(
                name="mlp_norm",
                op_subtype="mlp_norm",
                phase=_STATIC_PHASE,
                layer_idx=0,
                tokens=0,
                hidden=model_config.hidden_dim,
                ctx=ctx,
                count=count,
                tokens_fn=_num_tokens,
            ),
            GEMM(
                name="moe_gate",
                op_subtype="router",
                phase=_STATIC_PHASE,
                layer_idx=0,
                m=0,
                n=model_config.num_experts,
                k=model_config.hidden_dim,
                kernel_source="vllm_moe_gate",
                op_precision_override="fp32",
                ctx=ctx,
                count=count,
                m_fn=_num_tokens,
            ),
            ElementWise(
                name="moe_topk",
                op_subtype="topk",
                phase=_STATIC_PHASE,
                layer_idx=0,
                tokens=0,
                intermediate=model_config.num_experts,
                ctx=ctx,
                count=count,
                tokens_fn=_num_tokens,
            ),
            MoEDispatch(
                name="moe_dispatch_pre",
                pre_dispatch=True,
                phase=_STATIC_PHASE,
                layer_idx=0,
                num_tokens=0,
                hidden=model_config.hidden_dim,
                topk=model_config.num_activated_experts,
                num_experts=model_config.num_experts,
                ctx=ctx,
                count=count,
            ),
            MoE(
                name="routed_experts",
                op_subtype="routed_experts",
                phase=_STATIC_PHASE,
                layer_idx=rep_layer_idx,
                num_tokens=0,
                hidden=model_config.hidden_dim,
                moe_intermediate=model_config.expert_dim,
                topk=model_config.num_activated_experts,
                num_experts=model_config.num_experts,
                routing_distribution=routing.distribution,
                power_law_alpha=routing.power_law_alpha,
                skew=routing.get_skew_for_layer(rep_layer_idx),
                ctx=ctx,
                count=count,
            ),
            # post-dispatch 承载跨卡通信 (AIC 对齐): forward 按 tp/ep 解析为
            # allreduce(world=max(tp,ep)) 或本地. 不再有独立 routed_expert_allreduce op.
            MoEDispatch(
                name="moe_dispatch_post",
                pre_dispatch=False,
                phase=_STATIC_PHASE,
                layer_idx=0,
                num_tokens=0,
                hidden=model_config.hidden_dim,
                topk=model_config.num_activated_experts,
                num_experts=model_config.num_experts,
                ctx=ctx,
                count=count,
            ),
        ]

        if model_config.num_shared_experts > 0:
            shared_dim = _shared_dim_per_tp(ctx)
            ops.extend(
                [
                    GEMM(
                        name="shared_expert_up_gate",
                        op_subtype="shared_expert_up_gate",
                        phase=_STATIC_PHASE,
                        layer_idx=0,
                        m=0,
                        n=2 * shared_dim,
                        k=model_config.hidden_dim,
                        kernel_source="vllm_row_parallel_linear",
                        op_precision_override="bf16",
                        ctx=ctx,
                        count=count,
                        m_fn=_num_tokens,
                    ),
                    ElementWise(
                        name="shared_expert_act",
                        op_subtype="silu_mul",
                        phase=_STATIC_PHASE,
                        layer_idx=0,
                        tokens=0,
                        intermediate=shared_dim,
                        ctx=ctx,
                        count=count,
                        tokens_fn=_num_tokens,
                    ),
                    GEMM(
                        name="shared_expert_down",
                        op_subtype="shared_expert_down",
                        phase=_STATIC_PHASE,
                        layer_idx=0,
                        m=0,
                        n=model_config.hidden_dim,
                        k=shared_dim,
                        kernel_source="vllm_row_parallel_linear",
                        op_precision_override="bf16",
                        ctx=ctx,
                        count=count,
                        m_fn=_num_tokens,
                    ),
                ]
            )
            ops.append(
                AllReduce(
                    name="shared_expert_allreduce",
                    phase=_STATIC_PHASE,
                    layer_idx=0,
                    message_bytes=0,
                    world_size=ctx.tp_size,
                    ctx=ctx,
                    kernel_source=_TP_COLLECTIVE_KERNEL_SOURCE,
                    topology=_topology_hint(),
                    count=count,
                    message_bytes_fn=_message_bytes_hidden(ctx),
                )
            )

        ops.append(
            ElementWise(
                name="mlp_add",
                op_subtype="mlp_add",
                phase=_STATIC_PHASE,
                layer_idx=0,
                tokens=0,
                hidden=model_config.hidden_dim,
                ctx=ctx,
                count=count,
                tokens_fn=_num_tokens,
            )
        )
        return ops

    def _lm_head_ops(self) -> list[Operator]:
        model_config = self.model_config
        return [
            GEMM(
                name="lm_head",
                op_subtype="lm_head",
                phase=_STATIC_PHASE,
                layer_idx=None,
                m=0,
                n=model_config.vocab_size // self.ctx.tp_size,
                k=model_config.hidden_dim,
                kernel_source="vllm_row_parallel_linear",
                ctx=self.ctx,
                count=1,
                m_fn=_lm_head_m,
            )
        ]

    def forward(self, step: StepShape) -> StepOpPlan:
        return StepOpPlan(
            step_id=step.step_id,
            phase=step.phase,
            ops=self.ops,
            runtime=StepRuntime.from_step(step),
            metadata={
                "model": self.model_config.name,
                "arch": "qwen3_moe",
                "num_layers": self.model_config.num_layers,
                "num_experts": self.model_config.num_experts,
                "num_activated_experts": self.model_config.num_activated_experts,
                "execution_mode": step.execution_mode,
            },
        )


@register_model("Qwen3MoeForCausalLM")
def _build_qwen3_moe_graph(model, ctx, **_):
    return Qwen3MoeModel(model_config=model, ctx=ctx)
