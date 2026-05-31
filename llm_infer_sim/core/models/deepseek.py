"""DeepSeekModel — #158 Step 4 ctx-based, qwen.py 同款直接构造.

范围:
    DeepSeek V3 dense MLA layer (q_lora + kv_lora)
    DeepSeek V3 MoE FFN (跟 Qwen3-30B-A3B 同 op class)

V3.2 (sparse / lightning indexer) 与 V4 (sparse + HC + hash MoE) 均不支持 (已删除)。

#158 Step 4: 所有 dense ops 直接构造 (Embedding / Norm / ElementWise / GEMM /
Attention / FusedMoE / Collective), 不走 factory.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from llm_infer_sim.core.step.runtime import StepRuntime
from llm_infer_sim.core.step.step_plan import StepOpPlan
from llm_infer_sim.core.step.step_shape import StepShape
from llm_infer_sim.core.models.layer_partition import partition_ffn_layers
from llm_infer_sim.core.operators.context import OperatorContext
from llm_infer_sim.core.operators import (
    AllReduce,
    ElementWise,
    Embedding,
    GEMM,
    MLAAttention,
    MoE,
    MoEDispatch,
    MoERoutingProfile,
    Norm,
)
from llm_infer_sim.core.operators.base import Operator
from llm_infer_sim.core.models.registry import register_model
from llm_infer_sim.core.models.config import ModelProfile


def _mla_qk_head_dim(model: ModelProfile) -> int:
    qk_nope = model.qk_nope_head_dim if model.qk_nope_head_dim > 0 else model.head_dim
    return qk_nope + (model.rope_head_dim or 0)


def _mla_v_dim(model: ModelProfile) -> int:
    qk_nope = model.qk_nope_head_dim if model.qk_nope_head_dim > 0 else model.head_dim
    return model.v_head_dim if model.v_head_dim > 0 else qk_nope


def _mla_kv_latent_dim(model: ModelProfile) -> int:
    if model.kv_latent_dim > 0:
        return model.kv_latent_dim
    return model.kv_lora_rank + (model.rope_head_dim or 0)


def _shared_dim_per_tp(ctx: OperatorContext) -> int:
    return ctx.model.expert_dim * ctx.model.num_shared_experts // ctx.tp_size


def _comm_bytes_hidden(tokens: int, ctx: OperatorContext) -> int:
    return int(tokens * ctx.model.hidden_dim * ctx.a_byte)


@dataclass(frozen=True)
class DeepSeekModel:
    model: ModelProfile
    ctx: OperatorContext | None = None  # routing 随 ctx 传入 (ctx.routing)

    def _build_ops(self, step: StepShape):
        """(op, count, layer_indices) for the step, consumed by forward()."""
        if self.ctx is None:
            raise ValueError(
                "DeepSeekModel.ctx is required (engine builder should pass it)"
            )
        ctx = self.ctx
        m = self.model
        tokens = step.total_tokens
        phase = step.phase
        layer_count = m.num_layers
        all_layer_indices = tuple(range(layer_count))

        out: list[tuple] = [(
            Embedding(
                name="embedding", phase=phase, layer_idx=None, tokens=tokens,
                vocab_size=m.vocab_size, hidden=m.hidden_dim, ctx=ctx,
            ), 1, (),
        )]
        for op in self._build_mla_attn_block(0, step, ctx=ctx):
            out.append((op, layer_count, all_layer_indices))
        for ffn_kind, layer_indices in partition_ffn_layers(m):
            rep = layer_indices[0]
            ffn_ops = (self._build_dense_ffn_block(rep, step, ctx=ctx) if ffn_kind == "dense"
                       else self._build_moe_ffn_block(rep, step, ctx=ctx))
            for op in ffn_ops:
                out.append((op, len(layer_indices), layer_indices))
        head_tokens = (max(step.num_prefill_requests, 1) if phase == "prefill"
                       else step.num_decode_requests)
        out.append((
            GEMM(
                name="lm_head", op_subtype="lm_head", phase=phase, layer_idx=None,
                m=head_tokens, n=m.vocab_size // ctx.tp_size, k=m.hidden_dim,
                kernel_source="vllm_row_parallel_linear", ctx=ctx,
            ), 1, (),
        ))
        return out

    def _metadata(self, step: StepShape) -> dict:
        return {
            "model": self.model.name,
            "num_layers": self.model.num_layers,
            "execution_mode": step.execution_mode,
            "first_moe_layer": self.model.first_moe_layer,
        }

    def forward(self, step: StepShape) -> StepOpPlan:
        """Static-contract engine entry (op_plan §7). 与 Qwen 同路径:
        _build_ops → 带 count 的静态 op → StepOpPlan(runtime=...)。"""
        ops = tuple(
            dataclasses.replace(op, count=count)
            for op, count, _li in self._build_ops(step)
        )
        return StepOpPlan(
            step_id=step.step_id, phase=step.phase, ops=ops,
            runtime=StepRuntime.from_step(step), metadata=self._metadata(step),
        )

    # ---- MLA attention block (direct construction) ----

    def _build_mla_attn_block(
        self,
        layer_idx: int,
        step: StepShape,
        ctx: OperatorContext | None = None,
    ) -> list[Operator]:
        if ctx is None:
            ctx = self.ctx
        m = self.model
        tp = ctx.tp_size
        tokens = step.total_tokens
        phase = step.phase

        heads_per_tp = m.num_heads // tp
        qk_head_dim = _mla_qk_head_dim(m)
        v_dim = _mla_v_dim(m)
        qk_nope = m.qk_nope_head_dim if m.qk_nope_head_dim > 0 else m.head_dim
        qk_rope = m.rope_head_dim or 0
        prefix = f"layer{layer_idx}"

        ops: list[Operator] = [
            Norm(
                name="attn_norm", op_subtype="attn_norm",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, hidden=m.hidden_dim, ctx=ctx,
            ),
        ]

        # Q side: q_lora_rank > 0 → q_a_proj + q_b_proj, else single q_proj
        if m.q_lora_rank > 0:
            ops.append(GEMM(
                name=f"{prefix}_q_a_proj", op_subtype="q_a_proj",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=m.q_lora_rank, k=m.hidden_dim,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
            ))
            ops.append(GEMM(
                name=f"{prefix}_q_b_proj", op_subtype="q_b_proj",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=heads_per_tp * qk_head_dim, k=m.q_lora_rank,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
            ))
        else:
            ops.append(GEMM(
                name=f"{prefix}_q_proj", op_subtype="q_proj",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=heads_per_tp * qk_head_dim, k=m.hidden_dim,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
            ))

        # KV side: kv_a_proj_with_mqa (output -> KV cache) + kv_b_proj (compute-time decompression)
        ops.append(GEMM(
            name=f"{prefix}_kv_a_proj_with_mqa", op_subtype="kv_a_proj_with_mqa",
            phase=phase, layer_idx=layer_idx,
            m=tokens, n=m.kv_lora_rank + qk_rope, k=m.hidden_dim,
            kernel_source="vllm_row_parallel_linear",
            is_kv_proj=True,
            ctx=ctx,
        ))
        ops.append(GEMM(
            name=f"{prefix}_kv_b_proj", op_subtype="kv_b_proj",
            phase=phase, layer_idx=layer_idx,
            m=tokens, n=heads_per_tp * (qk_nope + v_dim), k=m.kv_lora_rank,
            kernel_source="vllm_row_parallel_linear",
            ctx=ctx,
        ))

        # V3 dense MLA attention (V3.2 sparse indexer 不支持)。
        ops.append(self._build_mla_attention(layer_idx, step, ctx))

        # O projection: heads_per_tp × v_head_dim → hidden (Row Parallel)
        ops.append(GEMM(
            name=f"{prefix}_o_proj", op_subtype="o_proj",
            phase=phase, layer_idx=layer_idx,
            m=tokens, n=m.hidden_dim, k=heads_per_tp * v_dim,
            kernel_source="vllm_row_parallel_linear",
            ctx=ctx,
        ))

        if tp > 1:
            ops.append(AllReduce(
                name="attn_allreduce",
                message_bytes=_comm_bytes_hidden(tokens, ctx),
                phase=phase, layer_idx=layer_idx, world_size=tp, ctx=ctx,
            ))
        ops.append(ElementWise(
            name="attn_add", op_subtype="attn_add",
            phase=phase, layer_idx=layer_idx,
            tokens=tokens, hidden=m.hidden_dim, ctx=ctx,
        ))
        return ops

    # ---- MLA attention op (V3 dense MLA) ----

    def _build_mla_attention(
        self,
        layer_idx: int,
        step: StepShape,
        ctx: OperatorContext,
    ) -> MLAAttention:
        m = self.model
        heads_per_tp = m.num_heads // ctx.tp_size
        qk_head_dim = _mla_qk_head_dim(m)
        v_dim = _mla_v_dim(m)
        kv_latent_dim = _mla_kv_latent_dim(m)

        if step.phase == "decode":
            return MLAAttention.mla_decode(
                layer_idx=layer_idx,
                ctx_len=step.avg_decode_context_len,
                bs=step.num_decode_requests,
                heads_per_tp=heads_per_tp,
                kv_latent_dim=kv_latent_dim,
                kv_lora_rank=m.kv_lora_rank,
                ctx=ctx,
            )
        if step.phase == "prefill":
            return MLAAttention.mla_prefill(
                layer_idx=layer_idx,
                seqlen=step.max_prefill_seqlen,
                bs=max(step.num_prefill_requests, 1),
                ctx_len=step.max_context_len,
                heads_per_tp=heads_per_tp,
                qk_head_dim=qk_head_dim, v_dim=v_dim,
                kv_latent_dim=kv_latent_dim,
                ctx=ctx,
            )
        raise NotImplementedError(
            f"mla_attention 只支持 prefill/decode, got {step.phase!r}"
        )

    # ---- dense FFN block ----

    def _build_dense_ffn_block(
        self,
        layer_idx: int,
        step: StepShape,
        ctx: OperatorContext | None = None,
    ) -> list[Operator]:
        if ctx is None:
            ctx = self.ctx
        tokens = step.total_tokens
        phase = step.phase
        m = self.model
        tp = ctx.tp_size
        inter_per_tp = m.ffn_dim // tp

        ops: list[Operator] = [
            Norm(
                name="mlp_norm", op_subtype="mlp_norm",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, hidden=m.hidden_dim, ctx=ctx,
            ),
            GEMM(
                name="gate_up_proj", op_subtype="gate_up_proj",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=2 * inter_per_tp, k=m.hidden_dim,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
            ),
            ElementWise(
                name="mlp_act", op_subtype="mlp_act",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, intermediate=inter_per_tp, ctx=ctx,
            ),
            GEMM(
                name="down_proj", op_subtype="down_proj",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=m.hidden_dim, k=inter_per_tp,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
            ),
        ]
        if tp > 1:
            ops.append(AllReduce(
                name="mlp_allreduce",
                message_bytes=_comm_bytes_hidden(tokens, ctx),
                phase=phase, layer_idx=layer_idx, world_size=tp, ctx=ctx,
            ))
        ops.append(ElementWise(
            name="mlp_add", op_subtype="mlp_add",
            phase=phase, layer_idx=layer_idx,
            tokens=tokens, hidden=m.hidden_dim, ctx=ctx,
        ))
        return ops

    # ---- MoE FFN block ----

    def _build_moe_ffn_block(
        self,
        layer_idx: int,
        step: StepShape,
        ctx: OperatorContext | None = None,
    ) -> list[Operator]:
        if ctx is None:
            ctx = self.ctx
        tokens = step.total_tokens
        phase = step.phase
        m = self.model
        tp = ctx.tp_size
        routing = ctx.routing or MoERoutingProfile.balanced()
        # vLLM 默认 MoE 路径: 本地 fused_experts + 单次 allreduce 聚合 partial sums
        # (不走 AllToAll). 通信由 post-dispatch op 承载 (AIC 对齐: MoEDispatch=通信).

        ops: list[Operator] = [
            Norm(
                name="mlp_norm", op_subtype="mlp_norm",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, hidden=m.hidden_dim, ctx=ctx,
            ),
            GEMM(
                name="moe_gate", op_subtype="router",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=m.num_experts, k=m.hidden_dim,
                kernel_source="vllm_moe_gate",
                op_precision_override="fp32",
                ctx=ctx,
            ),
            # moe_plan §3.3 op#3: softmax + topk kernel
            ElementWise(
                name="moe_topk", op_subtype="topk",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, intermediate=m.num_experts, ctx=ctx,
            ),
            # moe_plan §3.3 op#4: local pre-dispatch (vLLM fused_moe 内部 permute/align)
            MoEDispatch(
                name="moe_dispatch_pre", pre_dispatch=True,
                phase=phase, layer_idx=layer_idx, num_tokens=tokens,
                hidden=m.hidden_dim, topk=m.num_activated_experts,
                num_experts=m.num_experts, ctx=ctx,
            ),
            # moe_plan §3.3 op#6: routed experts (weight 已按 ep+tp 切, expert_map → partial sum)
            MoE(
                name="routed_experts", op_subtype="routed_experts",
                phase=phase, layer_idx=layer_idx, num_tokens=tokens,
                hidden=m.hidden_dim, moe_intermediate=m.expert_dim,
                topk=m.num_activated_experts, num_experts=m.num_experts,
                routing_distribution=routing.distribution,
                power_law_alpha=routing.power_law_alpha,
                skew=routing.get_skew_for_layer(layer_idx), ctx=ctx,
            ),
            # moe_plan §3.3 op#8: post-dispatch 承载跨卡通信 (forward 按 tp/ep 解析为
            #   allreduce(world=max(tp,ep)) 或本地); 不再有独立 routed_expert_allreduce.
            MoEDispatch(
                name="moe_dispatch_post", pre_dispatch=False,
                phase=phase, layer_idx=layer_idx, num_tokens=tokens,
                hidden=m.hidden_dim, topk=m.num_activated_experts,
                num_experts=m.num_experts, ctx=ctx,
            ),
        ]

        if m.num_shared_experts > 0:
            shared_per_tp = _shared_dim_per_tp(ctx)
            ops.append(GEMM(
                name="shared_expert_up_gate", op_subtype="shared_expert_up_gate",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=2 * shared_per_tp, k=m.hidden_dim,
                kernel_source="vllm_row_parallel_linear",
                op_precision_override="bf16",
                ctx=ctx,
            ))
            ops.append(ElementWise(
                name="shared_expert_act", op_subtype="silu_mul",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, intermediate=shared_per_tp, ctx=ctx,
            ))
            ops.append(GEMM(
                name="shared_expert_down", op_subtype="shared_expert_down",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=m.hidden_dim, k=shared_per_tp,
                kernel_source="vllm_row_parallel_linear",
                op_precision_override="bf16",
                ctx=ctx,
            ))
            if tp > 1:
                ops.append(AllReduce(
                    name="shared_expert_allreduce",
                    message_bytes=_comm_bytes_hidden(tokens, ctx),
                    phase=phase, layer_idx=layer_idx, world_size=tp, ctx=ctx,
                ))

        ops.append(ElementWise(
            name="mlp_add", op_subtype="mlp_add",
            phase=phase, layer_idx=layer_idx,
            tokens=tokens, hidden=m.hidden_dim, ctx=ctx,
        ))
        return ops


@register_model("DeepseekV3ForCausalLM")
def _build_deepseek_graph(model, ctx, **_):
    return DeepSeekModel(model=model, ctx=ctx)
