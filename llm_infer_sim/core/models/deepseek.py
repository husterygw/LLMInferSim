"""DeepSeekModelTemplate — V2/V3 MLA attention + dense (first_k) / MoE (rest) FFN.

阶段 3b 范围:
    DeepSeek V3 dense MLA layer (q_lora + kv_lora)
    DeepSeek V3 MoE FFN (跟 Qwen3-30B-A3B 同 MoEOpFactory)

未做 (后续阶段):
    V3.2 sparse attention (indexer)                — 3c
    V4 sparse attention + HC (hyper-connection)    — 3c
    FP4 / FP8 weight quantization                  — 后续
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.operators.factories import FactoryBundle
from llm_infer_sim.core.operators.specs import Operator
from llm_infer_sim.core.profiles.model_config import ModelConfig


def _ctx_len_from_step(step: StepShape) -> int:
    """Pick ctx_len for V3.2 indexer kernel (sees full attention context)."""
    if step.phase == "decode":
        return step.avg_decode_context_len
    return step.max_context_len if step.max_context_len > 0 else step.max_prefill_seqlen


@dataclass(frozen=True)
class DeepSeekModelTemplate:
    model: ModelConfig

    def build_step(
        self,
        step: StepShape,
        factories: FactoryBundle,
    ) -> StepOpPlan:
        ops: list[Operator] = []
        tokens = step.total_tokens
        phase = step.phase

        is_v4 = self.model.window_size > 0 and self.model.o_groups > 0

        ops.append(factories.embedding.embedding(tokens, phase))
        # V4 HC: embedding 后实化 hc_mult 副本
        if self.model.hc_mult > 0:
            ops.append(factories.embedding.hc_embedding_repeat(tokens, phase))

        for layer_idx in range(self.model.num_layers):
            if is_v4:
                ops.extend(self._build_v4_attn_block(layer_idx, step, factories))
            else:
                ops.extend(self._build_mla_attn_block(layer_idx, step, factories))

            if self.model.is_moe_layer(layer_idx):
                ops.extend(self._build_moe_ffn_block(layer_idx, step, factories))
            else:
                ops.extend(self._build_dense_ffn_block(layer_idx, step, factories))

        if phase == "prefill":
            head_tokens = max(step.num_prefill_requests, 1)
        else:
            head_tokens = step.num_decode_requests
        # V4 model-level HC head + final_norm 在 lm_head 之前
        if self.model.hc_mult > 0:
            ops.append(factories.embedding.hc_head(head_tokens, phase))
            ops.append(factories.embedding.final_norm(head_tokens, phase))
        ops.append(factories.dense.lm_head(head_tokens, phase))

        return StepOpPlan(
            step_id=step.step_id,
            phase=phase,
            ops=tuple(ops),
            metadata={
                "model": self.model.name,
                "num_layers": self.model.num_layers,
                "execution_mode": step.execution_mode,
                "first_moe_layer": self.model.first_moe_layer,
                "is_v4": is_v4,
            },
        )

    # ---- MLA attention block ----

    def _build_mla_attn_block(
        self,
        layer_idx: int,
        step: StepShape,
        factories: FactoryBundle,
    ) -> list[Operator]:
        if factories.collective is None:
            raise ValueError(
                "DeepSeekModelTemplate requires FactoryBundle.collective"
            )
        m = self.model
        deploy = factories.dense.deploy
        tp = deploy.tp_size
        tokens = step.total_tokens
        phase = step.phase

        heads_per_tp = m.num_heads // tp
        qk_nope = m.qk_nope_head_dim if m.qk_nope_head_dim > 0 else m.head_dim
        qk_rope = m.rope_head_dim or 0
        v_dim = m.v_head_dim if m.v_head_dim > 0 else qk_nope
        q_head_dim = qk_nope + qk_rope

        ops: list[Operator] = [factories.norm.attn_norm(layer_idx, tokens, phase)]

        prefix = f"layer{layer_idx}"

        # Q side: q_lora_rank > 0 → q_a_proj + q_b_proj, else single q_proj
        if m.q_lora_rank > 0:
            ops.append(factories.dense.linear(
                name=f"{prefix}_q_a_proj", op_subtype="q_a_proj",
                layer_idx=layer_idx, tokens=tokens, phase=phase,
                ic=m.hidden_dim, oc=m.q_lora_rank,
            ))
            ops.append(factories.dense.linear(
                name=f"{prefix}_q_b_proj", op_subtype="q_b_proj",
                layer_idx=layer_idx, tokens=tokens, phase=phase,
                ic=m.q_lora_rank, oc=heads_per_tp * q_head_dim,
            ))
        else:
            ops.append(factories.dense.linear(
                name=f"{prefix}_q_proj", op_subtype="q_proj",
                layer_idx=layer_idx, tokens=tokens, phase=phase,
                ic=m.hidden_dim, oc=heads_per_tp * q_head_dim,
            ))

        # KV side: kv_a_proj_with_mqa (output -> KV cache) + kv_b_proj (compute-time decompression)
        ops.append(factories.dense.linear(
            name=f"{prefix}_kv_a_proj_with_mqa", op_subtype="kv_a_proj_with_mqa",
            layer_idx=layer_idx, tokens=tokens, phase=phase,
            ic=m.hidden_dim, oc=m.kv_lora_rank + qk_rope,
            is_kv_proj=True,
        ))
        ops.append(factories.dense.linear(
            name=f"{prefix}_kv_b_proj", op_subtype="kv_b_proj",
            layer_idx=layer_idx, tokens=tokens, phase=phase,
            ic=m.kv_lora_rank, oc=heads_per_tp * (qk_nope + v_dim),
        ))

        # V3.2 DSA: lightning indexer + sparse MLA attention.
        # V3 (无 indexer): dense MLA attention.
        if m.index_topk > 0 and factories.indexer is not None:
            ctx_len = _ctx_len_from_step(step)
            ops.extend(factories.indexer.v32_indexer_block(
                layer_idx, tokens, phase, ctx_len=ctx_len,
            ))
            ops.append(factories.attention.mla_sparse_attention(layer_idx, step))
        else:
            ops.append(factories.attention.mla_attention(layer_idx, step))

        # O projection: heads_per_tp × v_head_dim → hidden  (Row Parallel)
        ops.append(factories.dense.linear(
            name=f"{prefix}_o_proj", op_subtype="o_proj",
            layer_idx=layer_idx, tokens=tokens, phase=phase,
            ic=heads_per_tp * v_dim, oc=m.hidden_dim,
        ))

        if tp > 1:
            comm_bytes = int(tokens * m.hidden_dim * factories.norm.a_byte)
            ops.append(factories.collective.allreduce(
                name="attn_allreduce",
                message_bytes=comm_bytes,
                phase=phase, layer_idx=layer_idx, world_size=tp,
            ))
        ops.append(factories.norm.attn_add(layer_idx, tokens, phase))
        return ops

    # ---- dense FFN block (V3 first_k layers) ----

    def _build_dense_ffn_block(
        self,
        layer_idx: int,
        step: StepShape,
        factories: FactoryBundle,
    ) -> list[Operator]:
        tokens = step.total_tokens
        phase = step.phase
        deploy = factories.dense.deploy
        tp = deploy.tp_size

        ops: list[Operator] = [
            factories.norm.mlp_norm(layer_idx, tokens, phase),
            factories.dense.gate_up_proj(layer_idx, tokens, phase),
            factories.norm.mlp_act(layer_idx, tokens, phase),
            factories.dense.down_proj(layer_idx, tokens, phase),
        ]
        if tp > 1 and factories.collective is not None:
            ops.append(factories.collective.allreduce(
                name="mlp_allreduce",
                message_bytes=int(tokens * self.model.hidden_dim * factories.norm.a_byte),
                phase=phase, layer_idx=layer_idx, world_size=tp,
            ))
        ops.append(factories.norm.mlp_add(layer_idx, tokens, phase))
        return ops

    # ---- MoE FFN block (reuse Qwen template's logic + V4 hash + HC) ----

    def _build_moe_ffn_block(
        self,
        layer_idx: int,
        step: StepShape,
        factories: FactoryBundle,
    ) -> list[Operator]:
        if factories.moe is None or factories.collective is None:
            raise ValueError(
                "MoE FFN requires FactoryBundle.moe + .collective"
            )
        tokens = step.total_tokens
        phase = step.phase
        deploy = factories.moe.deploy
        tp = deploy.tp_size
        ep = deploy.ep_size
        a_byte = factories.moe.a_byte
        comm_bytes_h = int(tokens * self.model.hidden_dim * a_byte)

        ops: list[Operator] = []
        # V4 HC pre (FFN side)
        if self.model.hc_mult > 0:
            ops.append(factories.norm.hc_pre(layer_idx, tokens, phase, scope="ffn"))

        ops.append(factories.norm.mlp_norm(layer_idx, tokens, phase))

        # V4 hash routing: 前 num_hash_layers 层用 tid2eid lookup 替代 moe_gate
        is_hash = (0 <= layer_idx < self.model.num_hash_layers
                   if self.model.num_hash_layers > 0 else False)
        if is_hash:
            ops.append(factories.moe.moe_hash_lookup(layer_idx, tokens, phase))
        else:
            ops.append(factories.moe.moe_gate(layer_idx, tokens, phase))

        if ep > 1:
            ops.append(factories.collective.alltoall(
                name="ep_alltoall_dispatch",
                message_bytes=comm_bytes_h,
                phase=phase, layer_idx=layer_idx, world_size=ep,
            ))
        ops.append(factories.moe.routed_experts(layer_idx, tokens, phase))
        if ep == 1 and tp > 1:
            ops.append(factories.collective.allreduce(
                name="routed_expert_allreduce",
                message_bytes=comm_bytes_h,
                phase=phase, layer_idx=layer_idx, world_size=tp,
            ))
        elif ep > 1:
            ops.append(factories.collective.alltoall(
                name="ep_alltoall_combine",
                message_bytes=comm_bytes_h,
                phase=phase, layer_idx=layer_idx, world_size=ep,
            ))

        if self.model.num_shared_experts > 0:
            ops.append(factories.moe.shared_expert_up_gate(layer_idx, tokens, phase))
            ops.append(factories.moe.shared_expert_act(layer_idx, tokens, phase))
            ops.append(factories.moe.shared_expert_down(layer_idx, tokens, phase))
            if tp > 1:
                ops.append(factories.collective.allreduce(
                    name="shared_expert_allreduce",
                    message_bytes=comm_bytes_h,
                    phase=phase, layer_idx=layer_idx, world_size=tp,
                ))

        # V4 HC post 替代 mlp_add; 否则常规 mlp_add
        if self.model.hc_mult > 0:
            ops.append(factories.norm.hc_post(layer_idx, tokens, phase, scope="ffn"))
        else:
            ops.append(factories.norm.mlp_add(layer_idx, tokens, phase))
        return ops

    # ---- V4 attention block (sparse + compressor + indexer + low-rank O) ----

    def _build_v4_attn_block(
        self,
        layer_idx: int,
        step: StepShape,
        factories: FactoryBundle,
    ) -> list[Operator]:
        if factories.v4_attention is None or factories.collective is None:
            raise ValueError(
                "V4 path requires FactoryBundle.v4_attention + .collective"
            )
        m = self.model
        deploy = factories.dense.deploy
        tp = deploy.tp_size
        tokens = step.total_tokens
        phase = step.phase
        ctx_len = _ctx_len_from_step(step)
        a_byte = factories.norm.a_byte
        compress_ratio = m.get_compress_ratio(layer_idx)

        ops: list[Operator] = []
        # HC pre (attention side)
        if m.hc_mult > 0:
            ops.append(factories.norm.hc_pre(layer_idx, tokens, phase, scope="attn"))

        ops.append(factories.norm.attn_norm(layer_idx, tokens, phase))
        ops.append(factories.v4_attention.fused_wqa_wkv(layer_idx, tokens, phase))
        ops.append(factories.v4_attention.q_norm(layer_idx, tokens, phase))
        ops.append(factories.v4_attention.wq_b(layer_idx, tokens, phase))
        ops.append(factories.v4_attention.kv_norm(layer_idx, tokens, phase))

        # Compressor (compress_ratio > 0)
        if compress_ratio > 0:
            ops.extend(factories.v4_attention.compressor_ops(
                layer_idx, tokens, ctx_len, phase, compress_ratio,
            ))
            # Indexer 只在 compress_ratio == 4 (CSA) 走
            if compress_ratio == 4 and factories.indexer is not None and m.index_topk > 0:
                op_prec = "fp8" if factories.dense.w_byte <= 1.0 else "fp16"
                ops.extend(factories.indexer.v4_indexer_ops(
                    layer_idx=layer_idx, tokens=tokens, ctx_len=ctx_len,
                    phase=phase, compress_ratio=compress_ratio, op_prec=op_prec,
                ))

        # Sparse attention
        ops.append(factories.v4_attention.sparse_attention(layer_idx, step, compress_ratio))

        # Low-rank O: wo_a + wo_b
        ops.append(factories.v4_attention.wo_a(layer_idx, tokens, phase))
        ops.append(factories.v4_attention.wo_b(layer_idx, tokens, phase))

        if tp > 1:
            ops.append(factories.collective.allreduce(
                name="attn_allreduce",
                message_bytes=int(tokens * m.hidden_dim * a_byte),
                phase=phase, layer_idx=layer_idx, world_size=tp,
            ))

        # HC post 替代 attn_add; 否则常规 attn_add
        if m.hc_mult > 0:
            ops.append(factories.norm.hc_post(layer_idx, tokens, phase, scope="attn"))
        else:
            ops.append(factories.norm.attn_add(layer_idx, tokens, phase))
        return ops
