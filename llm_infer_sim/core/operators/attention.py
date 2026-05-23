"""Attention operator — ctx-based with named constructors.

Attention 公式复杂 (FlashAttn tiling + KV cache layout), RooflineSpec 在
named constructor (Attention.flash_prefill / mla_decode / ...) 内部计算.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


@dataclass(frozen=True)
class Attention:
    """Attention op (regular GQA / MLA / sparse). RooflineSpec pre-computed in
    named constructor (Attention.flash_prefill / mla_decode / ...).
    """
    name: str
    op_subtype: str            # "prefill" / "decode" / "mixed_prefill" / "mixed_decode" / ...
    phase: str
    layer_idx: int | None
    roofline_spec_value: RooflineSpec
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    # signature shape inputs (OperatorDB key 用)
    num_tokens: int = 0
    num_seqs: int = 0
    q_len: int = 0
    kv_len: int = 0
    num_q_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    # attention-specific runtime keys
    attention_backend: str = ""
    kv_dtype: str = ""
    block_size: int = 0
    dtype_override: str | None = None
    kernel_source: str = "vllm_default"
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def op_kind(self) -> str:
        return "attention"

    @property
    def dtype(self) -> str:
        return self.dtype_override if self.dtype_override else self.ctx.dtype

    @property
    def shape(self) -> dict[str, Any]:
        return {
            "num_tokens": self.num_tokens,
            "num_seqs": self.num_seqs,
            "q_len": self.q_len,
            "kv_len": self.kv_len,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }

    @property
    def parallel(self) -> dict[str, Any]:
        return {"tp": self.ctx.tp_size}

    @property
    def runtime(self) -> dict[str, Any]:
        out = {
            "framework": self.ctx.framework,
            "framework_version": self.ctx.framework_version,
            "execution_mode": self.ctx.execution_mode,
            "kernel_source": self.kernel_source,
        }
        if self.attention_backend:
            out["attention_backend"] = self.attention_backend
        if self.kv_dtype:
            out["kv_dtype"] = self.kv_dtype
        if self.block_size:
            out["block_size"] = self.block_size
        return out

    def roofline_spec(self) -> RooflineSpec:
        return self.roofline_spec_value

    def signature(self) -> OperatorSignature:
        return OperatorSignature(
            op_kind="attention",
            op_subtype=self.op_subtype,
            dtype=self.dtype,
            shape=to_canonical(project(self.shape, (
                "num_tokens", "num_seqs", "q_len", "kv_len",
                "num_q_heads", "num_kv_heads", "head_dim",
            ))),
            parallel=to_canonical(project(self.parallel, ("tp",))),
            runtime=to_canonical(project(self.runtime, (
                "framework", "framework_version", "execution_mode",
                "kernel_source", "attention_backend", "kv_dtype", "block_size",
            ))),
        )

    # ---- FlashAttention (regular GQA) named constructors ----

    @classmethod
    def flash_prefill(
        cls, *,
        layer_idx: int, seqlen: int, bs: int,
        n_q: int, n_kv: int, head_dim: int,
        ctx: OperatorContext,
        phase: str = "prefill",
        name: str = "attention",
        op_subtype: str = "prefill",
        kernel_source: str = "vllm_flash_attn",
        tags: tuple[str, ...] = (),
    ) -> "Attention":
        """FlashAttention prefill: causal attention on seqlen × seqlen, per-batch.

        n_q / n_kv 已 / TP. flops = QK + SV + softmax; KV IO 按 FlashAttn tiling 估.
        """
        a_byte = ctx.a_byte
        kv_byte = ctx.kv_byte
        onchip = ctx.hw.onchip_buffer
        flops = (
            seqlen * seqlen * head_dim * n_q * bs * 2          # Q @ K^T
            + seqlen * head_dim * seqlen * n_q * bs * 2        # softmax @ V
            + bs * n_q * seqlen * seqlen * 5                   # softmax
        )
        block_size_r = min(math.ceil(onchip / (kv_byte * head_dim)), head_dim)
        n_blocks_r = math.ceil(seqlen / block_size_r)
        q_numel = seqlen * head_dim * bs * n_q * a_byte
        o_numel = seqlen * head_dim * bs * n_q * a_byte
        kv_io = int(n_blocks_r * seqlen * head_dim * bs * n_kv * kv_byte * 2)
        spec = RooflineSpec(
            op_category="attention",
            flops=int(flops),
            load_act=int(q_numel),
            store_act=int(o_numel * 2),
            load_kv_cache=kv_io,
        )
        return cls(
            name=name, op_subtype=op_subtype,
            phase=phase, layer_idx=layer_idx,
            num_tokens=bs * seqlen, num_seqs=bs,
            q_len=seqlen, kv_len=seqlen,
            num_q_heads=n_q, num_kv_heads=n_kv, head_dim=head_dim,
            attention_backend="flash_attn",
            kv_dtype="bf16",
            block_size=ctx.block_size,
            kernel_source=kernel_source,
            ctx=ctx, tags=tags,
            roofline_spec_value=spec,
        )

    @classmethod
    def flash_decode(
        cls, *,
        layer_idx: int, ctx_len: int, bs: int,
        n_q: int, n_kv: int, head_dim: int,
        ctx: OperatorContext,
        phase: str = "decode",
        name: str = "attention",
        op_subtype: str = "decode",
        kernel_source: str = "vllm_flash_attn",
        tags: tuple[str, ...] = (),
    ) -> "Attention":
        """FlashAttention decode: Q=1 token, KV=ctx_len from cache, per-batch."""
        a_byte = ctx.a_byte
        kv_byte = ctx.kv_byte
        onchip = ctx.hw.onchip_buffer
        flops = (
            ctx_len * head_dim * n_q * bs * 2
            + head_dim * ctx_len * n_q * bs * 2
            + bs * n_q * ctx_len * 5
        )
        block_size_r = min(math.ceil(onchip / (kv_byte * head_dim)), head_dim)
        n_blocks_r = math.ceil(1 / block_size_r)
        q_numel = head_dim * bs * n_q * a_byte
        o_numel = head_dim * bs * n_q * a_byte
        kv_io = int(n_blocks_r * ctx_len * head_dim * bs * n_kv * kv_byte * 2)
        spec = RooflineSpec(
            op_category="attention",
            flops=int(flops),
            load_act=int(q_numel),
            store_act=int(o_numel * 2),
            load_kv_cache=kv_io,
        )
        return cls(
            name=name, op_subtype=op_subtype,
            phase=phase, layer_idx=layer_idx,
            num_tokens=bs, num_seqs=bs,
            q_len=1, kv_len=ctx_len,
            num_q_heads=n_q, num_kv_heads=n_kv, head_dim=head_dim,
            attention_backend="flash_attn",
            kv_dtype="bf16",
            block_size=ctx.block_size,
            kernel_source=kernel_source,
            ctx=ctx, tags=tags,
            roofline_spec_value=spec,
        )

    # ---- MLA (DeepSeek V3) named constructors ----

    @classmethod
    def mla_prefill(
        cls, *,
        layer_idx: int, seqlen: int, bs: int, ctx_len: int,
        heads_per_tp: int, qk_head_dim: int, v_dim: int, kv_latent_dim: int,
        ctx: OperatorContext,
        phase: str = "prefill",
        name: str = "mla_attention",
        op_subtype: str = "prefill",
        kernel_source: str = "vllm_mla_prefill",
        attention_backend: str = "flash_attn_mla",
        tags: tuple[str, ...] = ("mla",),
    ) -> "Attention":
        """MLA prefill (FlashAttn-2 diff_headdims). New tokens read from staging act,
        prior context (chunked prefill) read from c_kv cache."""
        a_byte = ctx.a_byte
        kv_byte = ctx.kv_byte
        onchip = ctx.hw.onchip_buffer
        qk_ops = seqlen * seqlen * qk_head_dim * heads_per_tp * bs * 2
        sv_ops = seqlen * v_dim * seqlen * heads_per_tp * bs * 2
        softmax_ops = bs * heads_per_tp * seqlen * seqlen * 5

        block_size_r = min(math.ceil(onchip / (kv_byte * qk_head_dim)), qk_head_dim)
        n_blocks_r = math.ceil(seqlen / block_size_r)
        q_numel = seqlen * qk_head_dim * bs * heads_per_tp * a_byte
        o_numel = seqlen * v_dim * bs * heads_per_tp * a_byte
        kv_staging_act = int(
            n_blocks_r * seqlen * (qk_head_dim + v_dim) * bs * heads_per_tp * a_byte
        )
        prior_ctx = max(0, ctx_len - seqlen)
        kv_io = int(prior_ctx * kv_latent_dim * bs * kv_byte) if prior_ctx > 0 else 0
        spec = RooflineSpec(
            op_category="attention",
            flops=int(qk_ops + sv_ops + softmax_ops),
            load_act=int(q_numel + kv_staging_act),
            store_act=int(o_numel * 2),
            load_kv_cache=kv_io,
        )
        return cls(
            name=name, op_subtype=op_subtype,
            phase=phase, layer_idx=layer_idx,
            num_tokens=bs * seqlen, num_seqs=bs,
            q_len=seqlen, kv_len=seqlen + prior_ctx,
            num_q_heads=heads_per_tp, num_kv_heads=1,
            head_dim=kv_latent_dim,
            attention_backend=attention_backend,
            kv_dtype="bf16",
            block_size=ctx.block_size,
            kernel_source=kernel_source,
            ctx=ctx, tags=tags,
            roofline_spec_value=spec,
        )

    @classmethod
    def mla_decode(
        cls, *,
        layer_idx: int, ctx_len: int, bs: int,
        heads_per_tp: int, kv_latent_dim: int, kv_lora_rank: int,
        ctx: OperatorContext,
        phase: str = "decode",
        name: str = "mla_attention",
        op_subtype: str = "decode",
        kernel_source: str = "vllm_flashmla",
        attention_backend: str = "flashmla",
        tags: tuple[str, ...] = ("mla",),
    ) -> "Attention":
        """MLA decode (FlashMLA): K/V both from c_kv cache (single-head)."""
        a_byte = ctx.a_byte
        kv_byte = ctx.kv_byte
        onchip = ctx.hw.onchip_buffer
        _qk = kv_latent_dim
        _sv = kv_lora_rank or kv_latent_dim
        qk_ops = ctx_len * _qk * heads_per_tp * bs * 2
        sv_ops = 1 * _sv * ctx_len * heads_per_tp * bs * 2
        softmax_ops = bs * heads_per_tp * ctx_len * 1 * 5

        block_size_r = min(math.ceil(onchip / (kv_byte * _qk)), _qk)
        n_blocks_r = math.ceil(1 / block_size_r)
        q_numel = 1 * _qk * bs * heads_per_tp * a_byte
        o_numel = 1 * _sv * bs * heads_per_tp * a_byte
        kv_io = int(n_blocks_r * ctx_len * kv_latent_dim * bs * kv_byte)
        spec = RooflineSpec(
            op_category="attention",
            flops=int(qk_ops + sv_ops + softmax_ops),
            load_act=int(q_numel),
            store_act=int(o_numel * 2),
            load_kv_cache=kv_io,
        )
        return cls(
            name=name, op_subtype=op_subtype,
            phase=phase, layer_idx=layer_idx,
            num_tokens=bs, num_seqs=bs,
            q_len=1, kv_len=ctx_len,
            num_q_heads=heads_per_tp, num_kv_heads=1,
            head_dim=kv_latent_dim,
            attention_backend=attention_backend,
            kv_dtype="bf16",
            block_size=ctx.block_size,
            kernel_source=kernel_source,
            ctx=ctx, tags=tags,
            roofline_spec_value=spec,
        )

    # ---- MLA Sparse (DeepSeek V3.2) named constructors ----

    @classmethod
    def mla_sparse_prefill(
        cls, *,
        layer_idx: int, seqlen: int, bs: int, ctx_len: int,
        heads_per_tp: int, qk_head_dim: int, v_dim: int,
        kv_latent_dim: int, index_topk: int,
        ctx: OperatorContext,
        phase: str = "prefill",
        name: str = "mla_sparse_attention",
        op_subtype: str = "prefill",
        kernel_source: str = "vllm_mla_sparse_prefill",
        attention_backend: str = "flash_attn_mla_sparse",
        tags: tuple[str, ...] = ("mla", "sparse"),
    ) -> "Attention":
        """V3.2 sparse MLA prefill. attended_sum = seqlen × avg_attended."""
        a_byte = ctx.a_byte
        kv_byte = ctx.kv_byte
        onchip = ctx.hw.onchip_buffer
        prior_ctx = max(0, ctx_len - seqlen)
        avg_ctx = (seqlen + 1) / 2 + prior_ctx
        avg_attended = min(avg_ctx, float(index_topk)) if index_topk > 0 else avg_ctx
        attended_sum = int(seqlen * avg_attended)

        qk_ops = int(attended_sum * qk_head_dim * heads_per_tp * bs * 2)
        sv_ops = int(attended_sum * v_dim * heads_per_tp * bs * 2)
        softmax_ops = int(bs * heads_per_tp * attended_sum * 5)

        block_size_r = min(math.ceil(onchip / (kv_byte * qk_head_dim)), qk_head_dim)
        n_blocks_r = math.ceil(seqlen / block_size_r)
        q_numel = seqlen * qk_head_dim * bs * heads_per_tp * a_byte
        o_numel = seqlen * v_dim * bs * heads_per_tp * a_byte
        kv_staging_act = int(
            n_blocks_r * avg_attended * (qk_head_dim + v_dim) * bs * heads_per_tp * a_byte
        )
        if prior_ctx > 0:
            prior_attended = min(prior_ctx, index_topk) if index_topk > 0 else prior_ctx
            kv_io = int(prior_attended * kv_latent_dim * bs * kv_byte)
        else:
            kv_io = 0
        spec = RooflineSpec(
            op_category="attention",
            flops=int(qk_ops + sv_ops + softmax_ops),
            load_act=int(q_numel + kv_staging_act),
            store_act=int(o_numel * 2),
            load_kv_cache=kv_io,
        )
        return cls(
            name=name, op_subtype=op_subtype,
            phase=phase, layer_idx=layer_idx,
            num_tokens=bs * seqlen, num_seqs=bs,
            q_len=seqlen, kv_len=seqlen + prior_ctx,
            num_q_heads=heads_per_tp, num_kv_heads=1,
            head_dim=kv_latent_dim,
            attention_backend=attention_backend,
            kv_dtype="bf16",
            block_size=ctx.block_size,
            kernel_source=kernel_source,
            ctx=ctx, tags=tags,
            roofline_spec_value=spec,
        )

    @classmethod
    def mla_sparse_decode(
        cls, *,
        layer_idx: int, ctx_len: int, bs: int,
        heads_per_tp: int, kv_latent_dim: int,
        kv_lora_rank: int, index_topk: int,
        ctx: OperatorContext,
        phase: str = "decode",
        name: str = "mla_sparse_attention",
        op_subtype: str = "decode",
        kernel_source: str = "vllm_flashmla_sparse",
        attention_backend: str = "flashmla_sparse",
        tags: tuple[str, ...] = ("mla", "sparse"),
    ) -> "Attention":
        """V3.2 sparse MLA decode. attended <= min(ctx_len, index_topk)."""
        a_byte = ctx.a_byte
        kv_byte = ctx.kv_byte
        onchip = ctx.hw.onchip_buffer
        attended = min(ctx_len, index_topk) if index_topk > 0 else ctx_len
        _qk = kv_latent_dim
        _sv = kv_lora_rank or kv_latent_dim
        qk_ops = attended * _qk * heads_per_tp * bs * 2
        sv_ops = 1 * _sv * attended * heads_per_tp * bs * 2
        softmax_ops = bs * heads_per_tp * attended * 1 * 5

        block_size_r = min(math.ceil(onchip / (kv_byte * _qk)), _qk)
        n_blocks_r = math.ceil(1 / block_size_r)
        q_numel = 1 * _qk * bs * heads_per_tp * a_byte
        o_numel = 1 * _sv * bs * heads_per_tp * a_byte
        kv_io = int(n_blocks_r * attended * kv_latent_dim * bs * kv_byte)
        spec = RooflineSpec(
            op_category="attention",
            flops=int(qk_ops + sv_ops + softmax_ops),
            load_act=int(q_numel),
            store_act=int(o_numel * 2),
            load_kv_cache=kv_io,
        )
        return cls(
            name=name, op_subtype=op_subtype,
            phase=phase, layer_idx=layer_idx,
            num_tokens=bs, num_seqs=bs,
            q_len=1, kv_len=ctx_len,
            num_q_heads=heads_per_tp, num_kv_heads=1,
            head_dim=kv_latent_dim,
            attention_backend=attention_backend,
            kv_dtype="bf16",
            block_size=ctx.block_size,
            kernel_source=kernel_source,
            ctx=ctx, tags=tags,
            roofline_spec_value=spec,
        )
