"""MLA attention operator (DeepSeek V3 dense).

从 Attention 拆出 (op_plan §7 收尾): 普通 GQA/Flash 在 attention.py, MLA 在此。
MLAAttention 保持 per-regime 设计 (DeepSeek per-step 模板按 phase 建一个 mla op):
每个 named constructor 设 `spec_fn(q,kv,n)` 闭包 (捕获静态 MLA 维度), forward(step)
按 op_subtype 的 regime 解析 step shape, roofline_spec(op_runtime) = spec_fn 重算
(recompute==baked, 零漂移; DeepSeek 无 bench oracle, 严格锁等价)。

V3.2 sparse (lightning indexer) 不支持 (已删除)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_infer_sim.core.graph.runtime import OpRuntime, StepRuntime
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import OperatorBase, RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


# ---- MLA roofline formulas (pure; shape-driven) ----

def _mla_prefill_spec(
    *, seqlen: int, bs: int, ctx_len: int, heads_per_tp: int, qk_head_dim: int,
    v_dim: int, kv_latent_dim: int, a_byte: float, kv_byte: float, onchip: float,
) -> RooflineSpec:
    causal_factor = (seqlen + 1) / (2 * seqlen)
    qk_ops = seqlen * seqlen * qk_head_dim * heads_per_tp * bs * 2 * causal_factor
    sv_ops = seqlen * v_dim * seqlen * heads_per_tp * bs * 2 * causal_factor
    softmax_ops = bs * heads_per_tp * seqlen * seqlen * 5 * causal_factor
    block_size_r = min(math.ceil(onchip / (4 * kv_byte * qk_head_dim)), qk_head_dim)
    n_blocks_r = math.ceil(seqlen / block_size_r)
    q_numel = seqlen * qk_head_dim * bs * heads_per_tp * a_byte
    o_numel = seqlen * v_dim * bs * heads_per_tp * a_byte
    kv_causal_factor = (n_blocks_r + 1) / (2 * n_blocks_r)
    kv_staging_act = int(
        n_blocks_r * seqlen * (qk_head_dim + v_dim) * bs * heads_per_tp * a_byte
        * kv_causal_factor
    )
    prior_ctx = max(0, ctx_len - seqlen)
    kv_io = int(prior_ctx * kv_latent_dim * bs * kv_byte) if prior_ctx > 0 else 0
    return RooflineSpec(
        op_category="attention", flops=int(qk_ops + sv_ops + softmax_ops),
        load_act=int(q_numel + kv_staging_act), store_act=int(o_numel * 2),
        load_kv_cache=kv_io,
    )


def _mla_decode_spec(
    *, ctx_len: int, bs: int, heads_per_tp: int, kv_latent_dim: int,
    kv_lora_rank: int, a_byte: float, kv_byte: float, onchip: float,
) -> RooflineSpec:
    _qk = kv_latent_dim
    _sv = kv_lora_rank or kv_latent_dim
    qk_ops = ctx_len * _qk * heads_per_tp * bs * 2
    sv_ops = 1 * _sv * ctx_len * heads_per_tp * bs * 2
    softmax_ops = bs * heads_per_tp * ctx_len * 1 * 5
    block_size_r = min(math.ceil(onchip / (4 * kv_byte * _qk)), _qk)
    n_blocks_r = math.ceil(1 / block_size_r)
    q_numel = 1 * _qk * bs * heads_per_tp * a_byte
    o_numel = 1 * _sv * bs * heads_per_tp * a_byte
    kv_io = int(n_blocks_r * ctx_len * kv_latent_dim * bs * kv_byte)
    return RooflineSpec(
        op_category="attention", flops=int(qk_ops + sv_ops + softmax_ops),
        load_act=int(q_numel), store_act=int(o_numel * 2), load_kv_cache=kv_io,
    )


@dataclass(frozen=True)
class MLAAttention(OperatorBase):
    """MLA attention (DeepSeek V3 dense). Per-regime op (DeepSeek per-step 模板
    按 phase 建一个), spec_fn 从 step 重算 (recompute==baked)."""
    name: str
    op_subtype: str            # "prefill" / "decode" / "mixed_prefill" / "mixed_decode"
    phase: str
    layer_idx: int | None
    roofline_spec_value: RooflineSpec
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    num_tokens: int = 0
    num_seqs: int = 0
    q_len: int = 0
    kv_len: int = 0
    num_q_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    attention_backend: str = ""
    kv_dtype: str = ""
    block_size: int = 0
    dtype_override: str | None = None
    kernel_source: str = "vllm_default"
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ("mla",)
    count: int = 1
    spec_fn: Callable[[int, int, int], RooflineSpec] | None = field(
        default=None, compare=False, hash=False, repr=False,
    )

    @property
    def op_kind(self) -> str:
        return "attention"

    @property
    def shape(self) -> dict[str, Any]:
        return {
            "num_tokens": self.num_tokens, "num_seqs": self.num_seqs,
            "q_len": self.q_len, "kv_len": self.kv_len,
            "num_q_heads": self.num_q_heads, "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }

    # dtype / parallel ({"tp": tp_size}) 走 OperatorBase 默认。

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

    def forward(self, step: StepRuntime) -> OpRuntime | None:
        is_decode = "decode" in self.op_subtype
        mixed = step.phase in ("mixed", "chunked_prefill")
        if is_decode:
            if step.num_decode_requests <= 0:
                return None
            bs = step.num_decode_requests
            q_len, kv_len, num_tokens = 1, step.avg_decode_context_len, bs
            subtype = "mixed_decode" if mixed else "decode"
        else:
            if step.num_prefill_tokens <= 0:
                return None
            bs = max(step.num_prefill_requests, 1)
            seqlen = step.max_prefill_seqlen
            # MLA: kv_len = ctx_len (max_context_len else max_prefill_seqlen)
            kv_len = step.max_context_len if step.max_context_len > 0 else seqlen
            q_len, num_tokens = seqlen, bs * seqlen
            subtype = "mixed_prefill" if mixed else "prefill"
        shape = {
            "num_tokens": num_tokens, "num_seqs": bs, "q_len": q_len, "kv_len": kv_len,
            "num_q_heads": self.num_q_heads, "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }
        return OpRuntime(phase=step.phase, op_subtype=subtype, shape=shape,
                         parallel=dict(self.parallel), runtime=dict(self.runtime))

    def roofline_spec(self, op_runtime: OpRuntime | None = None) -> RooflineSpec:
        if op_runtime is None or self.spec_fn is None:
            return self.roofline_spec_value
        s = op_runtime.shape
        return self.spec_fn(int(s["q_len"]), int(s["kv_len"]), int(s["num_seqs"]))

    def signature(self, op_runtime: OpRuntime | None = None) -> OperatorSignature:
        return self.resolved_signature(
            op_runtime,
            shape_keys=(
                "num_tokens", "num_seqs", "q_len", "kv_len",
                "num_q_heads", "num_kv_heads", "head_dim",
            ),
            parallel_keys=("tp",),
            runtime_keys=(
                "framework", "framework_version", "execution_mode",
                "kernel_source", "attention_backend", "kv_dtype", "block_size",
            ),
        )

    # ---- MLA (DeepSeek V3 dense) named constructors ----

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
    ) -> "MLAAttention":
        prior_ctx = max(0, ctx_len - seqlen)

        def _spec(q: int, kv: int, n: int) -> RooflineSpec:
            return _mla_prefill_spec(
                seqlen=q, bs=n, ctx_len=kv, heads_per_tp=heads_per_tp,
                qk_head_dim=qk_head_dim, v_dim=v_dim, kv_latent_dim=kv_latent_dim,
                a_byte=ctx.a_byte, kv_byte=ctx.kv_byte, onchip=ctx.hw.onchip_buffer,
            )
        spec = _spec(seqlen, seqlen + prior_ctx, bs)
        return cls(
            name=name, op_subtype=op_subtype, phase=phase, layer_idx=layer_idx,
            num_tokens=bs * seqlen, num_seqs=bs, q_len=seqlen, kv_len=seqlen + prior_ctx,
            num_q_heads=heads_per_tp, num_kv_heads=1, head_dim=kv_latent_dim,
            attention_backend=attention_backend, kv_dtype="bf16",
            block_size=ctx.block_size, kernel_source=kernel_source,
            ctx=ctx, tags=tags, roofline_spec_value=spec, spec_fn=_spec,
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
    ) -> "MLAAttention":
        def _spec(q: int, kv: int, n: int) -> RooflineSpec:
            return _mla_decode_spec(
                ctx_len=kv, bs=n, heads_per_tp=heads_per_tp,
                kv_latent_dim=kv_latent_dim, kv_lora_rank=kv_lora_rank,
                a_byte=ctx.a_byte, kv_byte=ctx.kv_byte, onchip=ctx.hw.onchip_buffer,
            )
        spec = _spec(1, ctx_len, bs)
        return cls(
            name=name, op_subtype=op_subtype, phase=phase, layer_idx=layer_idx,
            num_tokens=bs, num_seqs=bs, q_len=1, kv_len=ctx_len,
            num_q_heads=heads_per_tp, num_kv_heads=1, head_dim=kv_latent_dim,
            attention_backend=attention_backend, kv_dtype="bf16",
            block_size=ctx.block_size, kernel_source=kernel_source,
            ctx=ctx, tags=tags, roofline_spec_value=spec, spec_fn=_spec,
        )

