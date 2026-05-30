"""Qwen3 dense model flat builder.

This follows the aiconfigurator model style: the model class builds a static op
list in ``__init__`` and ``forward`` only attaches step runtime. Qwen3-MoE
is intentionally not handled here; it belongs in ``qwen3_moe.py``.
"""
from __future__ import annotations

import os

from llm_infer_sim.core.graph.runtime import StepRuntime
from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.operators import (
    AllReduce,
    Attention,
    ElementWise,
    Embedding,
    GEMM,
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


class Qwen3Model:
    """Dense Qwen3 op graph.

    The static model structure is built once. Runtime-dependent fields such as
    token count, attention regime, and lm_head batch are resolved by each op's
    ``forward(StepRuntime)`` method.
    """

    def __init__(
        self,
        model_config: ModelProfile | None = None,
        ctx: OperatorContext | None = None,
        *,
        model: ModelProfile | None = None,
    ) -> None:
        if model_config is None:
            model_config = model
        if model_config is None or ctx is None:
            raise TypeError("Qwen3Model requires model_config and ctx")
        if model_config.num_experts > 0:
            raise ValueError(
                "Qwen3Model only supports dense Qwen3; use Qwen3MoeModel"
            )
        self.model_config = model_config
        self.ctx = ctx
        self.ops: tuple[Operator, ...] = tuple(self._build_ops())

    def _build_ops(self) -> list[Operator]:
        ops: list[Operator] = []
        ops.extend(self._embedding_ops())
        ops.extend(self._decoder_layer_ops(count=self.model_config.num_layers))
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

    def _decoder_layer_ops(self, *, count: int) -> list[Operator]:
        model_config = self.model_config
        ctx = self.ctx
        tp = ctx.tp_size
        n_q = model_config.num_heads // tp
        n_kv = model_config.num_kv_heads // tp
        qkv_out = (n_q + 2 * n_kv) * model_config.head_dim
        q_out = n_q * model_config.head_dim
        ffn = model_config.ffn_dim // tp

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
        ops.extend(
            [
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
                ),
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
        )
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
                "arch": "qwen3",
                "num_layers": self.model_config.num_layers,
                "execution_mode": step.execution_mode,
            },
        )


@register_model("Qwen3ForCausalLM")
def _build_qwen3_graph(model, ctx, **_):
    return Qwen3Model(model_config=model, ctx=ctx)
