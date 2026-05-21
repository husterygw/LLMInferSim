"""Attention Operator factory."""
from __future__ import annotations

import dataclasses
import math

from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.operators.factories.common import dense_parallel, make_runtime
from llm_infer_sim.core.operators.ops import AttentionOp, ElementwiseOp
from llm_infer_sim.core.operators.specs import OperatorFormula
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig


def _replace_step(step: StepShape, *, phase: str) -> StepShape:
    """frozen StepShape clone with new phase (StepShape is frozen dataclass)."""
    return dataclasses.replace(step, phase=phase)


def _rename_op(op: AttentionOp, name: str, *, subtype: str) -> AttentionOp:
    return dataclasses.replace(op, name=name, op_subtype=subtype,
                                tags=tuple(set(op.tags) | {"mixed"}))


class AttentionOpFactory:
    def __init__(
        self,
        model: ModelConfig,
        deploy: DeployConfig,
        hw: HardwareConfig,
        *,
        a_byte: float = 2.0,
        kv_byte: float = 2.0,
    ):
        self.model = model
        self.deploy = deploy
        self.hw = hw
        self.a_byte = a_byte
        self.kv_byte = kv_byte

    def rope(self, layer_idx: int, tokens: int, phase: str) -> ElementwiseOp:
        tp = self.deploy.tp_size
        n_q = self.model.num_heads // tp
        n_kv = self.model.num_kv_heads // tp
        qk_elements = tokens * (n_q + n_kv) * self.model.head_dim
        return ElementwiseOp(
            name=f"layer{layer_idx}_rope",
            op_kind="elementwise",
            op_subtype="rope",
            phase=phase,
            layer_idx=layer_idx,
            dtype="bf16",
            shape_fields={
                "tokens": tokens,
                "num_q_heads": n_q,
                "num_kv_heads": n_kv,
                "head_dim": self.model.head_dim,
            },
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="activation",
                flops=qk_elements * 3,
                load_act=int(qk_elements * self.a_byte),
                store_act=int(qk_elements * self.a_byte),
            ),
        )

    def attention(self, layer_idx: int, step: StepShape) -> AttentionOp:
        tp = self.deploy.tp_size
        n_q = self.model.num_heads // tp
        n_kv = self.model.num_kv_heads // tp
        head_dim = self.model.head_dim

        if step.phase == "prefill":
            seqlen = step.max_prefill_seqlen
            bs = max(step.num_prefill_requests, 1)
            subtype = "prefill"
            q_len = seqlen
            kv_len = seqlen
            num_tokens = bs * seqlen
            num_seqs = bs
            flops = (
                seqlen * seqlen * head_dim * n_q * bs * 2
                + seqlen * head_dim * seqlen * n_q * bs * 2
                + bs * n_q * seqlen * seqlen * 5
            )
            block_size_r = min(
                math.ceil(self.hw.onchip_buffer / (self.kv_byte * head_dim)),
                head_dim,
            )
            n_blocks_r = math.ceil(seqlen / block_size_r)
            q_numel = seqlen * head_dim * bs * n_q * self.a_byte
            o_numel = seqlen * head_dim * bs * n_q * self.a_byte
            kv_io = int(n_blocks_r * seqlen * head_dim * bs * n_kv * self.kv_byte * 2)
        elif step.phase == "decode":
            ctx_len = step.avg_decode_context_len
            bs = step.num_decode_requests
            subtype = "decode"
            q_len = 1
            kv_len = ctx_len
            num_tokens = bs
            num_seqs = bs
            flops = (
                ctx_len * head_dim * n_q * bs * 2
                + head_dim * ctx_len * n_q * bs * 2
                + bs * n_q * ctx_len * 5
            )
            block_size_r = min(
                math.ceil(self.hw.onchip_buffer / (self.kv_byte * head_dim)),
                head_dim,
            )
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = head_dim * bs * n_q * self.a_byte
            o_numel = head_dim * bs * n_q * self.a_byte
            kv_io = int(n_blocks_r * ctx_len * head_dim * bs * n_kv * self.kv_byte * 2)
        else:
            raise NotImplementedError(
                f"AttentionOpFactory.attention only supports prefill/decode, got {step.phase!r}; "
                "for mixed step use mixed_attention()."
            )

        return AttentionOp(
            name=f"layer{layer_idx}_attention",
            op_kind="attention",
            op_subtype=subtype,
            phase=step.phase,
            layer_idx=layer_idx,
            dtype="bf16",
            shape_fields={
                "num_tokens": num_tokens,
                "num_seqs": num_seqs,
                "q_len": q_len,
                "kv_len": kv_len,
                "num_q_heads": n_q,
                "num_kv_heads": n_kv,
                "head_dim": head_dim,
            },
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields={
                **make_runtime(self.deploy, kernel_source="vllm_flash_attn"),
                "attention_backend": "flash_attn",
                "kv_dtype": "bf16",
                "block_size": self.deploy.block_size,
            },
            formula_value=OperatorFormula(
                op_category="attention",
                flops=int(flops),
                load_act=int(q_numel),
                store_act=int(o_numel * 2),
                load_kv_cache=kv_io,
            ),
        )

    def mixed_attention(
        self, layer_idx: int, step: StepShape,
    ) -> list[AttentionOp]:
        """Mixed step (prefill 段 + decode 段) → split-kernels: 返 2 个 AttentionOp.

        阶段 3d 仅实现 split_kernels (V3 §4.7.1b). unified_ragged 推到 Stage 6 ModuleProfile.
        """
        if step.phase not in ("mixed", "chunked_prefill"):
            raise ValueError(
                f"mixed_attention expects phase=mixed/chunked_prefill, got {step.phase!r}"
            )
        ops: list[AttentionOp] = []
        if step.num_prefill_tokens > 0:
            pf_step = _replace_step(step, phase="prefill")
            pf = self.attention(layer_idx, pf_step)
            ops.append(_rename_op(pf, f"layer{layer_idx}_attention_prefill",
                                  subtype="mixed_prefill"))
        if step.num_decode_requests > 0:
            dc_step = _replace_step(step, phase="decode")
            dc = self.attention(layer_idx, dc_step)
            ops.append(_rename_op(dc, f"layer{layer_idx}_attention_decode",
                                  subtype="mixed_decode"))
        return ops

    def mla_attention(self, layer_idx: int, step: StepShape) -> AttentionOp:
        """MLA fused attention (DeepSeek V2/V3, FlashMLA decode + FlashAttn-2 diff_headdims prefill).

        decode: K/V 来自 c_kv cache (kv_latent_dim single-head, 不切 num_kv_heads).
        prefill: 新 token K/V 来自 kv_b_proj staging activation (走 load_act, a_byte);
                 chunked prefill 跨 step 时 prior context 才读 c_kv (kv_latent_dim).
        """
        m = self.model
        tp = self.deploy.tp_size
        heads_per_tp = m.num_heads // tp

        qk_nope = m.qk_nope_head_dim if m.qk_nope_head_dim > 0 else m.head_dim
        qk_rope = m.rope_head_dim or 0
        v_dim = m.v_head_dim if m.v_head_dim > 0 else qk_nope
        qk_head_dim = qk_nope + qk_rope
        kv_latent_dim = m.kv_latent_dim if m.kv_latent_dim > 0 else (m.kv_lora_rank + qk_rope)

        if step.phase == "decode":
            ctx_len = step.avg_decode_context_len
            bs = step.num_decode_requests
            _qk = kv_latent_dim
            _sv = m.kv_lora_rank or kv_latent_dim
            qk_ops = ctx_len * _qk * heads_per_tp * bs * 2
            sv_ops = 1 * _sv * ctx_len * heads_per_tp * bs * 2
            softmax_ops = bs * heads_per_tp * ctx_len * 1 * 5

            block_size_r = min(
                math.ceil(self.hw.onchip_buffer / (self.kv_byte * _qk)),
                _qk,
            )
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = 1 * _qk * bs * heads_per_tp * self.a_byte
            o_numel = 1 * _sv * bs * heads_per_tp * self.a_byte
            kv_io = int(n_blocks_r * ctx_len * kv_latent_dim * bs * self.kv_byte)

            subtype = "decode"
            num_tokens = bs
            num_seqs = bs
            q_len = 1
            kv_len = ctx_len
            kernel_source = "vllm_flashmla"
            attention_backend = "flashmla"
            flops = qk_ops + sv_ops + softmax_ops
            load_act = int(q_numel)
            store_act = int(o_numel * 2)
        elif step.phase == "prefill":
            seqlen = step.max_prefill_seqlen
            bs = max(step.num_prefill_requests, 1)
            ctx_len = step.max_context_len
            qk_ops = seqlen * seqlen * qk_head_dim * heads_per_tp * bs * 2
            sv_ops = seqlen * v_dim * seqlen * heads_per_tp * bs * 2
            softmax_ops = bs * heads_per_tp * seqlen * seqlen * 5

            block_size_r = min(
                math.ceil(self.hw.onchip_buffer / (self.kv_byte * qk_head_dim)),
                qk_head_dim,
            )
            n_blocks_r = math.ceil(seqlen / block_size_r)
            q_numel = seqlen * qk_head_dim * bs * heads_per_tp * self.a_byte
            o_numel = seqlen * v_dim * bs * heads_per_tp * self.a_byte
            kv_staging_act = int(
                n_blocks_r * seqlen * (qk_head_dim + v_dim) * bs * heads_per_tp * self.a_byte
            )
            prior_ctx = max(0, ctx_len - seqlen)
            kv_io = (
                int(prior_ctx * kv_latent_dim * bs * self.kv_byte) if prior_ctx > 0 else 0
            )

            subtype = "prefill"
            num_tokens = bs * seqlen
            num_seqs = bs
            q_len = seqlen
            kv_len = seqlen + prior_ctx
            kernel_source = "vllm_mla_prefill"
            attention_backend = "flash_attn_mla"
            flops = qk_ops + sv_ops + softmax_ops
            load_act = int(q_numel + kv_staging_act)
            store_act = int(o_numel * 2)
        else:
            raise NotImplementedError(
                f"mla_attention 只支持 prefill/decode, got {step.phase!r}"
            )

        return AttentionOp(
            name=f"layer{layer_idx}_mla_attention",
            op_kind="attention",
            op_subtype=subtype,
            phase=step.phase,
            layer_idx=layer_idx,
            dtype="bf16",
            tags=("mla",),
            shape_fields={
                "num_tokens": num_tokens,
                "num_seqs": num_seqs,
                "q_len": q_len,
                "kv_len": kv_len,
                "num_q_heads": heads_per_tp,
                "num_kv_heads": 1,                  # MLA c_kv single-head
                "head_dim": kv_latent_dim,
                "qk_head_dim": qk_head_dim,
                "v_head_dim": v_dim,
                "kv_lora_rank": m.kv_lora_rank,
            },
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields={
                **make_runtime(self.deploy, kernel_source=kernel_source),
                "attention_backend": attention_backend,
                "kv_dtype": "bf16",
                "block_size": self.deploy.block_size,
            },
            formula_value=OperatorFormula(
                op_category="attention",
                flops=int(flops),
                load_act=load_act,
                store_act=store_act,
                load_kv_cache=kv_io,
            ),
        )

    def mla_sparse_attention(self, layer_idx: int, step: StepShape) -> AttentionOp:
        """V3.2 DSA sparse MLA attention. attended_len 受 model.index_topk 截断."""
        m = self.model
        tp = self.deploy.tp_size
        heads_per_tp = m.num_heads // tp

        qk_nope = m.qk_nope_head_dim if m.qk_nope_head_dim > 0 else m.head_dim
        qk_rope = m.rope_head_dim or 0
        v_dim = m.v_head_dim if m.v_head_dim > 0 else qk_nope
        qk_head_dim = qk_nope + qk_rope
        kv_latent_dim = m.kv_latent_dim if m.kv_latent_dim > 0 else (m.kv_lora_rank + qk_rope)
        index_topk = m.index_topk

        if step.phase == "decode":
            ctx_len = step.avg_decode_context_len
            bs = step.num_decode_requests
            attended = min(ctx_len, index_topk) if index_topk > 0 else ctx_len
            _qk = kv_latent_dim
            _sv = m.kv_lora_rank or kv_latent_dim
            qk_ops = attended * _qk * heads_per_tp * bs * 2
            sv_ops = 1 * _sv * attended * heads_per_tp * bs * 2
            softmax_ops = bs * heads_per_tp * attended * 1 * 5

            block_size_r = min(
                math.ceil(self.hw.onchip_buffer / (self.kv_byte * _qk)),
                _qk,
            )
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = 1 * _qk * bs * heads_per_tp * self.a_byte
            o_numel = 1 * _sv * bs * heads_per_tp * self.a_byte
            kv_io = int(n_blocks_r * attended * kv_latent_dim * bs * self.kv_byte)

            subtype = "decode"
            num_tokens = bs
            num_seqs = bs
            q_len = 1
            kv_len = ctx_len
            attended_len = attended
            flops = qk_ops + sv_ops + softmax_ops
            load_act = int(q_numel)
            store_act = int(o_numel * 2)
            kernel_source = "vllm_flashmla_sparse"
            attention_backend = "flashmla_sparse"
        elif step.phase == "prefill":
            seqlen = step.max_prefill_seqlen
            bs = max(step.num_prefill_requests, 1)
            ctx_len = step.max_context_len
            prior_ctx = max(0, ctx_len - seqlen)
            avg_ctx = (seqlen + 1) / 2 + prior_ctx
            avg_attended = min(avg_ctx, float(index_topk)) if index_topk > 0 else avg_ctx
            attended_sum = int(seqlen * avg_attended)

            qk_ops = int(attended_sum * qk_head_dim * heads_per_tp * bs * 2)
            sv_ops = int(attended_sum * v_dim * heads_per_tp * bs * 2)
            softmax_ops = int(bs * heads_per_tp * attended_sum * 5)

            block_size_r = min(
                math.ceil(self.hw.onchip_buffer / (self.kv_byte * qk_head_dim)),
                qk_head_dim,
            )
            n_blocks_r = math.ceil(seqlen / block_size_r)
            q_numel = seqlen * qk_head_dim * bs * heads_per_tp * self.a_byte
            o_numel = seqlen * v_dim * bs * heads_per_tp * self.a_byte
            kv_staging_act = int(
                n_blocks_r * avg_attended * (qk_head_dim + v_dim) * bs * heads_per_tp * self.a_byte
            )
            if prior_ctx > 0:
                prior_attended = min(prior_ctx, index_topk) if index_topk > 0 else prior_ctx
                kv_io = int(prior_attended * kv_latent_dim * bs * self.kv_byte)
            else:
                kv_io = 0

            subtype = "prefill"
            num_tokens = bs * seqlen
            num_seqs = bs
            q_len = seqlen
            kv_len = seqlen + prior_ctx
            attended_len = int(avg_attended)
            flops = qk_ops + sv_ops + softmax_ops
            load_act = int(q_numel + kv_staging_act)
            store_act = int(o_numel * 2)
            kernel_source = "vllm_mla_sparse_prefill"
            attention_backend = "flash_attn_mla_sparse"
        else:
            raise NotImplementedError(
                f"mla_sparse_attention 只支持 prefill/decode, got {step.phase!r}"
            )

        return AttentionOp(
            name=f"layer{layer_idx}_mla_sparse_attention",
            op_kind="attention",
            op_subtype=subtype,
            phase=step.phase,
            layer_idx=layer_idx,
            dtype="bf16",
            tags=("mla", "sparse"),
            shape_fields={
                "num_tokens": num_tokens,
                "num_seqs": num_seqs,
                "q_len": q_len,
                "kv_len": kv_len,
                "attended_len": attended_len,
                "num_q_heads": heads_per_tp,
                "num_kv_heads": 1,
                "head_dim": kv_latent_dim,
                "qk_head_dim": qk_head_dim,
                "v_head_dim": v_dim,
                "kv_lora_rank": m.kv_lora_rank,
                "index_topk": index_topk,
            },
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields={
                **make_runtime(self.deploy, kernel_source=kernel_source),
                "attention_backend": attention_backend,
                "kv_dtype": "bf16",
                "block_size": self.deploy.block_size,
            },
            formula_value=OperatorFormula(
                op_category="attention",
                flops=int(flops),
                load_act=load_act,
                store_act=store_act,
                load_kv_cache=kv_io,
            ),
        )
