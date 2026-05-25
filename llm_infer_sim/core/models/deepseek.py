"""DeepSeekModelTemplate — #158 Step 4 ctx-based, qwen.py 同款直接构造.

阶段 3b 范围:
    DeepSeek V3 dense MLA layer (q_lora + kv_lora)
    DeepSeek V3 MoE FFN (跟 Qwen3-30B-A3B 同 op class)

阶段 3c V3.2:
    V3.2 sparse attention (lightning indexer) — 所有 layer attention 同结构, grouped 可用

V4 (sparse + HC + hash MoE): 已从此 template 删除 (#157).

#158 Step 4: 所有 dense ops 直接构造 (Embedding / Norm / ElementWise / GEMM /
Attention / FusedMoE / Collective), 不走 factory.
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.graph.grouped_plan import GroupedOperator, GroupedStepPlan
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.models.layer_partition import partition_ffn_layers
from llm_infer_sim.core.operators.context import OperatorContext
from llm_infer_sim.core.operators import (
    AllReduce,
    AllToAll,
    Attention,
    ElementWise,
    Embedding,
    FusedMoE,
    GEMM,
    MoERoutingProfile,
    Norm,
    RooflineOperator,
)
from llm_infer_sim.core.operators.base import Operator
from llm_infer_sim.core.profiles.model_config import ModelConfig


def _ctx_len_from_step(step: StepShape) -> int:
    """Pick ctx_len for MLA / V3.2 indexer kernel (sees full attention context)."""
    if step.phase == "decode":
        return step.avg_decode_context_len
    return step.max_context_len if step.max_context_len > 0 else step.max_prefill_seqlen


def _mla_qk_head_dim(model: ModelConfig) -> int:
    qk_nope = model.qk_nope_head_dim if model.qk_nope_head_dim > 0 else model.head_dim
    return qk_nope + (model.rope_head_dim or 0)


def _mla_v_dim(model: ModelConfig) -> int:
    qk_nope = model.qk_nope_head_dim if model.qk_nope_head_dim > 0 else model.head_dim
    return model.v_head_dim if model.v_head_dim > 0 else qk_nope


def _mla_kv_latent_dim(model: ModelConfig) -> int:
    if model.kv_latent_dim > 0:
        return model.kv_latent_dim
    return model.kv_lora_rank + (model.rope_head_dim or 0)


def _shared_dim_per_tp(ctx: OperatorContext) -> int:
    return ctx.model.expert_dim * ctx.model.num_shared_experts // ctx.tp_size


def _comm_bytes_hidden(tokens: int, ctx: OperatorContext) -> int:
    return int(tokens * ctx.model.hidden_dim * ctx.a_byte)


@dataclass(frozen=True)
class DeepSeekModelTemplate:
    model: ModelConfig
    ctx: OperatorContext | None = None
    routing: MoERoutingProfile | None = None
    indexer_kv_byte: float = 1.0

    def build_grouped_step(self, step: StepShape) -> GroupedStepPlan:
        """V3 dense MLA / V3.2 sparse grouped path."""
        if self.ctx is None:
            raise ValueError(
                "DeepSeekModelTemplate.ctx is required (engine builder should pass it)"
            )
        ctx = self.ctx
        m = self.model
        tokens = step.total_tokens
        phase = step.phase
        layer_count = m.num_layers
        all_layer_indices = tuple(range(layer_count))

        groups: list[GroupedOperator] = [
            GroupedOperator(
                op=Embedding(
                    name="embedding",
                    phase=phase, layer_idx=None,
                    tokens=tokens,
                    vocab_size=m.vocab_size,
                    hidden=m.hidden_dim,
                    ctx=ctx,
                ),
                count=1, layer_indices=(),
            ),
        ]

        # MLA / sparse-MLA attention block: layer-uniform
        for op in self._build_mla_attn_block(0, step, ctx=ctx):
            groups.append(GroupedOperator(
                op=op, count=layer_count, layer_indices=all_layer_indices,
            ))

        # FFN partition (dense first_moe_layer 层, MoE 其余)
        for ffn_kind, layer_indices in partition_ffn_layers(m):
            rep = layer_indices[0]
            if ffn_kind == "dense":
                ffn_ops = self._build_dense_ffn_block(rep, step, ctx=ctx)
            else:
                ffn_ops = self._build_moe_ffn_block(rep, step, ctx=ctx)
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
                n=m.vocab_size // ctx.tp_size,
                k=m.hidden_dim,
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
                "model": m.name,
                "num_layers": layer_count,
                "execution_mode": step.execution_mode,
                "first_moe_layer": m.first_moe_layer,
            },
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

        # V3.2 DSA: lightning indexer + sparse MLA attention.
        # V3 (无 indexer): dense MLA attention.
        if m.index_topk > 0:
            ctx_len = _ctx_len_from_step(step)
            ops.extend(self._build_v32_indexer_block(layer_idx, tokens, phase, ctx_len, ctx))
            ops.append(self._build_mla_sparse_attention(layer_idx, step, ctx))
        else:
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
    ) -> Attention:
        m = self.model
        heads_per_tp = m.num_heads // ctx.tp_size
        qk_head_dim = _mla_qk_head_dim(m)
        v_dim = _mla_v_dim(m)
        kv_latent_dim = _mla_kv_latent_dim(m)

        if step.phase == "decode":
            return Attention.mla_decode(
                layer_idx=layer_idx,
                ctx_len=step.avg_decode_context_len,
                bs=step.num_decode_requests,
                heads_per_tp=heads_per_tp,
                kv_latent_dim=kv_latent_dim,
                kv_lora_rank=m.kv_lora_rank,
                ctx=ctx,
            )
        if step.phase == "prefill":
            return Attention.mla_prefill(
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

    def _build_mla_sparse_attention(
        self,
        layer_idx: int,
        step: StepShape,
        ctx: OperatorContext,
    ) -> Attention:
        m = self.model
        heads_per_tp = m.num_heads // ctx.tp_size
        qk_head_dim = _mla_qk_head_dim(m)
        v_dim = _mla_v_dim(m)
        kv_latent_dim = _mla_kv_latent_dim(m)

        if step.phase == "decode":
            return Attention.mla_sparse_decode(
                layer_idx=layer_idx,
                ctx_len=step.avg_decode_context_len,
                bs=step.num_decode_requests,
                heads_per_tp=heads_per_tp,
                kv_latent_dim=kv_latent_dim,
                kv_lora_rank=m.kv_lora_rank,
                index_topk=m.index_topk,
                ctx=ctx,
            )
        if step.phase == "prefill":
            return Attention.mla_sparse_prefill(
                layer_idx=layer_idx,
                seqlen=step.max_prefill_seqlen,
                bs=max(step.num_prefill_requests, 1),
                ctx_len=step.max_context_len,
                heads_per_tp=heads_per_tp,
                qk_head_dim=qk_head_dim, v_dim=v_dim,
                kv_latent_dim=kv_latent_dim,
                index_topk=m.index_topk,
                ctx=ctx,
            )
        raise NotImplementedError(
            f"mla_sparse_attention 只支持 prefill/decode, got {step.phase!r}"
        )

    # ---- V3.2 lightning indexer 5-op block ----

    def _build_v32_indexer_block(
        self,
        layer_idx: int,
        tokens: int,
        phase: str,
        ctx_len: int,
        ctx: OperatorContext,
    ) -> list[Operator]:
        """V3.2 indexer: indexer_wq_b → indexer_wk_weights_proj → indexer_k_norm
        → indexer_q_fp8_quant → sparse_attn_indexer.

        indexer_k_norm + indexer_q_fp8_quant 用 RooflineOperator (legacy) 因为 Norm/
        ElementWise 的 hidden = head_dim (非 model.hidden_dim) 且 q_fp8_quant 的
        IO 不在 ElementWise standard formula 里; 留 legacy 直到 ElementWise 公式
        支持 quantize subtype.
        """
        from llm_infer_sim.core.operators.base import RooflineSpec

        # 内联 dense_parallel / make_runtime (factories/common.py 已删, #158 Step 5)
        def _dense_parallel(deploy):
            return {"tp": deploy.tp_size}

        def _make_runtime(deploy, *, kernel_source="vllm_default"):
            return {
                "framework": deploy.backend,
                "framework_version": deploy.backend_version or "unknown",
                "execution_mode": deploy.execution_mode,
                "kernel_source": kernel_source,
            }

        m = self.model
        h = m.hidden_dim
        n_head = m.index_n_heads
        head_dim = m.index_head_dim
        op_prec_idx = "fp8" if ctx.w_byte <= 1.0 else "fp16"
        a_byte = ctx.a_byte
        prefix = f"layer{layer_idx}"

        ops: list[Operator] = []

        # indexer_wq_b: q_lora_rank → n_head × head_dim (ReplicatedLinear, no /tp)
        ops.append(GEMM(
            name=f"{prefix}_indexer_wq_b", op_subtype="indexer_wq_b",
            phase=phase, layer_idx=layer_idx,
            m=tokens, n=n_head * head_dim, k=m.q_lora_rank,
            kernel_source="vllm_replicated_linear",
            op_precision_override=op_prec_idx,
            ctx=ctx,
        ))

        # indexer_wk_weights_proj: hidden → (head_dim + n_head), bf16 unquant
        fused_oc = head_dim + n_head
        ops.append(GEMM(
            name=f"{prefix}_indexer_wk_weights_proj", op_subtype="indexer_wk_weights_proj",
            phase=phase, layer_idx=layer_idx,
            m=tokens, n=fused_oc, k=h,
            kernel_source="vllm_replicated_linear",
            op_precision_override="bf16",
            weight_bytes_per_elem=2.0,    # 强制 bf16 weight (unquant)
            ctx=ctx,
        ))

        # indexer_k_norm: LayerNorm on head_dim (hidden=head_dim, 不是 model.hidden_dim)
        ops.append(Norm(
            name=f"{prefix}_indexer_k_norm", op_subtype="rmsnorm",
            phase=phase, layer_idx=layer_idx,
            tokens=tokens, hidden=head_dim, ctx=ctx,
        ))

        # indexer_q_fp8_quant: per-token group fp8 quant on Q. 用 RooflineOperator
        # legacy (ElementWise standard subtype 没 "quantize", IO 不规则).
        q_size = tokens * n_head * head_dim
        ops.append(RooflineOperator(
            name=f"{prefix}_indexer_q_fp8_quant",
            op_kind="elementwise", op_subtype="quantize",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"tokens": tokens, "n_head": n_head, "head_dim": head_dim},
            parallel_fields=_dense_parallel(ctx.deploy),
            runtime_fields=_make_runtime(ctx.deploy),
            roofline_spec_value=RooflineSpec(
                op_category="activation",
                flops=q_size * 5,
                load_act=int(q_size * a_byte),
                store_act=int(q_size * 1.0 + q_size // 128 * 4),
            ),
        ))

        # sparse_attn_indexer: Q×K_cache → top-k. Attention-kind 但有非标 shape (ctx_len/n_head).
        # 用 RooflineOperator legacy (Attention class shape 是 num_tokens/num_seqs/q_len/kv_len 定型).
        scale_bytes_per_pos = 4 * (head_dim // 128)
        ops.append(RooflineOperator(
            name=f"{prefix}_sparse_attn_indexer",
            op_kind="attention", op_subtype="sparse_index",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            tags=("v32_indexer",),
            shape_fields={
                "tokens": tokens, "ctx_len": ctx_len,
                "n_head": n_head, "head_dim": head_dim,
                "index_topk": m.index_topk,
            },
            parallel_fields=_dense_parallel(ctx.deploy),
            runtime_fields=_make_runtime(ctx.deploy, kernel_source="vllm_sparse_attn_indexer"),
            roofline_spec_value=RooflineSpec(
                op_category="attention",
                flops=tokens * ctx_len * head_dim * n_head * 2,
                load_act=int(tokens * n_head * head_dim * 1.0),
                load_kv_cache=int(ctx_len * (head_dim * self.indexer_kv_byte + scale_bytes_per_pos)),
                store_act=int(tokens * m.index_topk * 4),
            ),
        ))
        return ops

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
        ep = ctx.ep_size
        routing = self.routing or MoERoutingProfile.balanced()
        comm_bytes_h = _comm_bytes_hidden(tokens, ctx)

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
        ]

        if ep > 1:
            ops.append(AllToAll(
                name="ep_alltoall_dispatch",
                message_bytes=comm_bytes_h,
                phase=phase, layer_idx=layer_idx, world_size=ep, ctx=ctx,
            ))

        ops.append(FusedMoE.routed_experts(
            layer_idx=layer_idx, tokens=tokens, ctx=ctx, routing=routing, phase=phase,
        ))

        if ep == 1 and tp > 1:
            ops.append(AllReduce(
                name="routed_expert_allreduce",
                message_bytes=comm_bytes_h,
                phase=phase, layer_idx=layer_idx, world_size=tp, ctx=ctx,
            ))
        elif ep > 1:
            ops.append(AllToAll(
                name="ep_alltoall_combine",
                message_bytes=comm_bytes_h,
                phase=phase, layer_idx=layer_idx, world_size=ep, ctx=ctx,
            ))

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
                    message_bytes=comm_bytes_h,
                    phase=phase, layer_idx=layer_idx, world_size=tp, ctx=ctx,
                ))

        ops.append(ElementWise(
            name="mlp_add", op_subtype="mlp_add",
            phase=phase, layer_idx=layer_idx,
            tokens=tokens, hidden=m.hidden_dim, ctx=ctx,
        ))
        return ops
