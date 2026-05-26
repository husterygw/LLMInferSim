"""QwenModelGraphTemplate — #158 ctx-based.

阶段 1: Qwen3 dense TP=1, 每层:
    attn_norm → qkv_proj → rope → attention → o_proj → attn_add
        → mlp_norm → gate_up_proj → mlp_act → down_proj → mlp_add

TP>1 时 dense path 在 o_proj / down_proj 后追加 row-parallel allreduce:
    tp_o_proj_allreduce / tp_down_proj_allreduce

阶段 3a + moe_plan Phase 3 + vLLM 默认 EP path fix: Qwen3 MoE (e.g. Qwen3-30B-A3B):
    mlp_norm → moe_gate (router GEMM) → moe_topk (softmax + topk)
        → moe_dispatch_pre (local reorder/pack, vLLM fused_moe 内部 kernel)
        → routed_experts (MoE op, expert_map 处理 non-local → partial sum)
        → moe_dispatch_post (local gather/unpack)
        → [max(tp,ep) > 1: routed_expert_allreduce 聚合 partial sums]
        → [num_shared_experts>0: shared_expert_up_gate / _act / _down / _allreduce]
        → mlp_add

vLLM 默认 EP path (`enable_expert_parallel=True` + vllm_fused_moe Triton): 不走
AllToAll dispatch/combine, 而是 `fused_experts(expert_map=[-1 for non-local])` +
单次 `tensor_model_parallel_all_reduce`. TRT-LLM SM≥100 / SGLang DeepEP 才 AllToAll.

全图: embedding → [layer ops × num_layers] → lm_head

mixed/chunked_prefill phase 在 _attention_ops 内部拆 2 个 Attention op
(prefill segment + decode segment), 仍是 layer-uniform 不影响 grouping.

所有 op (Embedding / Norm / ElementWise / GEMM / Attention / MoE / MoEDispatch /
Collective) 由 template 直接构造, ctx 从 self.ctx 派生.
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.graph.grouped_plan import GroupedOperator, GroupedStepPlan
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.models.layer_partition import partition_ffn_layers
from llm_infer_sim.core.operators.context import OperatorContext
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
from llm_infer_sim.core.profiles.model_config import ModelConfig


# ---- Qwen dense shape helpers (private) ----

def _qkv_dim(ctx: OperatorContext) -> int:
    """Fused QKV projection output dim per-TP: (n_q + 2*n_kv) × head_dim."""
    m = ctx.model
    tp = ctx.tp_size
    n_q = m.num_heads // tp
    n_kv = m.num_kv_heads // tp
    return (n_q + 2 * n_kv) * m.head_dim


def _q_dim(ctx: OperatorContext) -> int:
    """Q projection dim per-TP: n_q × head_dim (= input dim of o_proj)."""
    m = ctx.model
    tp = ctx.tp_size
    n_q = m.num_heads // tp
    return n_q * m.head_dim


def _ffn_dim_per_tp(ctx: OperatorContext) -> int:
    """Dense FFN intermediate dim per-TP."""
    return ctx.model.ffn_dim // ctx.tp_size


def _lm_head_dim(ctx: OperatorContext) -> int:
    """LM head output dim per-TP (vocab sharded by TP)."""
    return ctx.model.vocab_size // ctx.tp_size


def _shared_dim_per_tp(ctx: OperatorContext) -> int:
    """Shared experts intermediate dim per-TP."""
    m = ctx.model
    return m.expert_dim * m.num_shared_experts // ctx.tp_size


def _comm_bytes_hidden(tokens: int, ctx: OperatorContext) -> int:
    """Activation bytes for hidden-state communication."""
    return int(tokens * ctx.model.hidden_dim * ctx.a_byte)


@dataclass(frozen=True)
class QwenModelGraphTemplate:
    model: ModelConfig
    ctx: OperatorContext | None = None
    routing: MoERoutingProfile | None = None

    def build_grouped_step(self, step: StepShape) -> GroupedStepPlan:
        """Grouped step plan. 用 self.ctx (engine builder 传入)."""
        if self.ctx is None:
            raise ValueError(
                "QwenModelGraphTemplate.ctx is required (engine builder should pass it)"
            )
        ctx = self.ctx
        tokens = step.total_tokens
        phase = step.phase
        layer_count = self.model.num_layers
        all_layer_indices = tuple(range(layer_count))

        groups: list[GroupedOperator] = [
            GroupedOperator(
                op=Embedding(
                    name="embedding",
                    phase=phase, layer_idx=None,
                    tokens=tokens,
                    vocab_size=self.model.vocab_size,
                    hidden=self.model.hidden_dim,
                    ctx=ctx,
                ),
                count=1, layer_indices=(),
            ),
        ]

        # attention block: layer-uniform
        for op in self._build_attn_block(0, step, ctx):
            groups.append(GroupedOperator(
                op=op, count=layer_count, layer_indices=all_layer_indices,
            ))

        # FFN block: (dense / moe) partition
        for ffn_kind, layer_indices in partition_ffn_layers(self.model):
            rep = layer_indices[0]
            if ffn_kind == "dense":
                ffn_ops = self._build_dense_ffn_block(rep, step, ctx)
            else:
                ffn_ops = self._build_moe_ffn_block(rep, step, ctx)
            for op in ffn_ops:
                groups.append(GroupedOperator(
                    op=op,
                    count=len(layer_indices),
                    layer_indices=layer_indices,
                ))

        if phase == "prefill":
            head_tokens = max(step.num_prefill_requests, 1)
        else:
            head_tokens = step.num_decode_requests
        groups.append(GroupedOperator(
            op=GEMM(
                name="lm_head", op_subtype="lm_head",
                phase=phase, layer_idx=None,
                m=head_tokens,
                n=_lm_head_dim(ctx),
                k=self.model.hidden_dim,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
            ),
            count=1, layer_indices=(),
        ))

        return GroupedStepPlan(
            step_id=step.step_id,
            phase=phase,
            groups=tuple(groups),
            metadata={
                "model": self.model.name,
                "num_layers": layer_count,
                "execution_mode": step.execution_mode,
            },
        )

    # ---- dense attention block (direct construction) ----

    def _attention_ops(self, layer_idx, step, ctx):
        """prefill/decode 单 op; mixed/chunked 拆 2 ops (prefill segment + decode segment)."""
        if step.phase in ("mixed", "chunked_prefill"):
            ops: list[Attention] = []
            if step.num_prefill_tokens > 0:
                ops.append(self._build_prefill_attention(
                    layer_idx, step.max_prefill_seqlen,
                    max(step.num_prefill_requests, 1),
                    "mixed", ctx,
                    name="attention_prefill", op_subtype="mixed_prefill",
                    extra_tags=("mixed",),
                ))
            if step.num_decode_requests > 0:
                ops.append(self._build_decode_attention(
                    layer_idx, step.avg_decode_context_len,
                    step.num_decode_requests,
                    "mixed", ctx,
                    name="attention_decode", op_subtype="mixed_decode",
                    extra_tags=("mixed",),
                ))
            return ops
        if step.phase == "prefill":
            return [self._build_prefill_attention(
                layer_idx, step.max_prefill_seqlen,
                max(step.num_prefill_requests, 1),
                step.phase, ctx,
            )]
        return [self._build_decode_attention(
            layer_idx, step.avg_decode_context_len,
            step.num_decode_requests,
            step.phase, ctx,
        )]

    def _build_prefill_attention(
        self,
        layer_idx: int, seqlen: int, bs: int, phase: str,
        ctx: OperatorContext, *,
        name: str = "attention", op_subtype: str = "prefill",
        extra_tags: tuple = (),
    ) -> Attention:
        tp = ctx.tp_size
        return Attention.flash_prefill(
            layer_idx=layer_idx, seqlen=seqlen, bs=bs,
            n_q=self.model.num_heads // tp,
            n_kv=self.model.num_kv_heads // tp,
            head_dim=self.model.head_dim,
            ctx=ctx, phase=phase,
            name=name, op_subtype=op_subtype, tags=extra_tags,
        )

    def _build_decode_attention(
        self,
        layer_idx: int, ctx_len: int, bs: int, phase: str,
        ctx: OperatorContext, *,
        name: str = "attention", op_subtype: str = "decode",
        extra_tags: tuple = (),
    ) -> Attention:
        tp = ctx.tp_size
        return Attention.flash_decode(
            layer_idx=layer_idx, ctx_len=ctx_len, bs=bs,
            n_q=self.model.num_heads // tp,
            n_kv=self.model.num_kv_heads // tp,
            head_dim=self.model.head_dim,
            ctx=ctx, phase=phase,
            name=name, op_subtype=op_subtype, tags=extra_tags,
        )

    def _build_attn_block(
        self,
        layer_idx: int,
        step: StepShape,
        ctx: OperatorContext,
    ) -> list[Operator]:
        """attn_norm → qkv → rope → attention → o → [tp_ar] → attn_add."""
        tokens = step.total_tokens
        phase = step.phase
        tp = ctx.tp_size
        n_q = self.model.num_heads // tp
        n_kv = self.model.num_kv_heads // tp
        ops: list[Operator] = [
            Norm(
                name="attn_norm", op_subtype="attn_norm",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, hidden=self.model.hidden_dim, ctx=ctx,
            ),
            GEMM(
                name="qkv_proj", op_subtype="qkv_proj",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=_qkv_dim(ctx), k=self.model.hidden_dim,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
            ),
            ElementWise(
                name="rope", op_subtype="rope",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens,
                num_heads=n_q + n_kv, head_dim=self.model.head_dim,
                ctx=ctx,
            ),
            *self._attention_ops(layer_idx, step, ctx),
            GEMM(
                name="o_proj", op_subtype="o_proj",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=self.model.hidden_dim, k=_q_dim(ctx),
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
            ),
        ]
        if tp > 1:
            ops.append(AllReduce(
                name="tp_o_proj_allreduce",
                message_bytes=_comm_bytes_hidden(tokens, ctx),
                phase=phase,
                layer_idx=layer_idx,
                world_size=tp,
                ctx=ctx,
            ))
        ops.append(
            ElementWise(
                name="attn_add", op_subtype="attn_add",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, hidden=self.model.hidden_dim, ctx=ctx,
            ),
        )
        return ops

    def _build_dense_ffn_block(
        self,
        layer_idx: int,
        step: StepShape,
        ctx: OperatorContext,
    ) -> list[Operator]:
        """mlp_norm → gate_up → act → down → [tp_ar] → mlp_add."""
        tokens = step.total_tokens
        phase = step.phase
        tp = ctx.tp_size
        ops: list[Operator] = [
            Norm(
                name="mlp_norm", op_subtype="mlp_norm",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, hidden=self.model.hidden_dim, ctx=ctx,
            ),
            GEMM(
                name="gate_up_proj", op_subtype="gate_up_proj",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=2 * _ffn_dim_per_tp(ctx), k=self.model.hidden_dim,
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
            ),
            ElementWise(
                name="mlp_act", op_subtype="mlp_act",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, intermediate=_ffn_dim_per_tp(ctx), ctx=ctx,
            ),
            GEMM(
                name="down_proj", op_subtype="down_proj",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=self.model.hidden_dim, k=_ffn_dim_per_tp(ctx),
                kernel_source="vllm_row_parallel_linear",
                ctx=ctx,
            ),
        ]
        if tp > 1:
            ops.append(AllReduce(
                name="tp_down_proj_allreduce",
                message_bytes=_comm_bytes_hidden(tokens, ctx),
                phase=phase,
                layer_idx=layer_idx,
                world_size=tp,
                ctx=ctx,
            ))
        ops.append(
            ElementWise(
                name="mlp_add", op_subtype="mlp_add",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, hidden=self.model.hidden_dim, ctx=ctx,
            ),
        )
        return ops

    # ---- per-layer composition helpers (for tests / introspection) ----

    def _build_layer(self, layer_idx: int, step: StepShape) -> list[Operator]:
        """Dense layer = attention block + dense FFN block."""
        ctx = self.ctx
        return [
            *self._build_attn_block(layer_idx, step, ctx),
            *self._build_dense_ffn_block(layer_idx, step, ctx),
        ]

    def _build_moe_layer(self, layer_idx: int, step: StepShape) -> list[Operator]:
        """MoE layer = attention block + MoE FFN block."""
        ctx = self.ctx
        return [
            *self._build_attn_block(layer_idx, step, ctx),
            *self._build_moe_ffn_block(layer_idx, step, ctx),
        ]

    # ---- MoE FFN block ----

    def _build_moe_ffn_block(
        self,
        layer_idx: int,
        step: StepShape,
        ctx: OperatorContext,
    ) -> list[Operator]:
        """moe_plan §3.3 + §3.4.1 标准化 MoE FFN block.

        TP-only (ep == 1):
          mlp_norm → moe_gate → moe_topk → moe_dispatch_pre → routed_experts
            → moe_dispatch_post → [tp>1: routed_expert_allreduce]

        EP (ep > 1):
          mlp_norm → moe_gate → moe_topk → moe_dispatch_pre → ep_alltoall_dispatch
            → routed_experts → ep_alltoall_combine → moe_dispatch_post
        """
        tokens = step.total_tokens
        phase = step.phase
        routing = self.routing or MoERoutingProfile.balanced()
        m = self.model
        tp = ctx.tp_size
        ep = ctx.ep_size
        # vLLM 默认 MoE EP path (`enable_expert_parallel=True` + vllm_fused_moe Triton):
        #   - weight 切分: ep>1 时切 expert 集合 (intermediate 不切 TP); ep=1 tp>1 时切 intermediate
        #   - 通信: fused_experts(expert_map=[-1 for non-local]) 产 partial sum,
        #          然后单次 tensor_model_parallel_all_reduce 跨 max(tp,ep) ranks 聚合
        #   - 不走 AllToAll dispatch/combine. 仅 TRT-LLM SM≥100 / SGLang DeepEP 才 AllToAll
        # MoEDispatch (pre/post) 仍存在, 表达 vLLM fused_moe 内部 local align/sort/gather kernel
        # (不替代 communication).
        ar_world_size = max(tp, ep)
        comm_bytes_tp = _comm_bytes_hidden(tokens, ctx)
        post_comm_peer = (
            "routed_expert_allreduce" if ar_world_size > 1 else None
        )

        ops: list[Operator] = [
            Norm(
                name="mlp_norm", op_subtype="mlp_norm",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, hidden=m.hidden_dim, ctx=ctx,
            ),
            # router GEMM (fp32 precision for routing stability)
            GEMM(
                name="moe_gate", op_subtype="router",
                phase=phase, layer_idx=layer_idx,
                m=tokens, n=m.num_experts, k=m.hidden_dim,
                kernel_source="vllm_moe_gate",
                op_precision_override="fp32",
                ctx=ctx,
            ),
            # moe_plan §3.3 op#3: 显式 softmax + topk kernel (placeholder elementwise)
            ElementWise(
                name="moe_topk", op_subtype="topk",
                phase=phase, layer_idx=layer_idx,
                tokens=tokens, intermediate=m.num_experts, ctx=ctx,
            ),
            # moe_plan §3.3 op#4: local dispatch (token reorder/pack, vLLM fused_moe 内部 kernel)
            MoEDispatch.local(
                pre_dispatch=True, layer_idx=layer_idx, tokens=tokens,
                ctx=ctx, phase=phase, communication_peer=None,
            ),
            # moe_plan §3.3 op#6: routed experts (MoE, backend-neutral)
            #   weight 已按 ep + tp 切, expert_map 处理 non-local experts → partial sum 输出
            MoE.routed_experts(
                layer_idx=layer_idx, tokens=tokens, ctx=ctx, routing=routing, phase=phase,
            ),
            # moe_plan §3.3 op#8: local combine (expert output gather/unpack)
            MoEDispatch.local(
                pre_dispatch=False, layer_idx=layer_idx, tokens=tokens,
                ctx=ctx, phase=phase, communication_peer=post_comm_peer,
            ),
        ]

        # vLLM 默认 path 单 AllReduce: tp>1 或 ep>1 时跨 max(tp,ep) ranks 聚合 partial sums.
        # 等价于 `tensor_model_parallel_all_reduce(reduce_results=True)`.
        if ar_world_size > 1:
            ops.append(AllReduce(
                name="routed_expert_allreduce",
                message_bytes=comm_bytes_tp,
                phase=phase, layer_idx=layer_idx,
                world_size=ar_world_size, ctx=ctx,
            ))

        # shared experts (V3 / Qwen3-Coder; A3B 没有)
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
                    message_bytes=comm_bytes_tp,
                    phase=phase, layer_idx=layer_idx, world_size=tp, ctx=ctx,
                ))

        ops.append(ElementWise(
            name="mlp_add", op_subtype="mlp_add",
            phase=phase, layer_idx=layer_idx,
            tokens=tokens, hidden=m.hidden_dim, ctx=ctx,
        ))
        return ops
