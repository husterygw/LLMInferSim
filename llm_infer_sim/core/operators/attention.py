"""Attention operator (FlashAttention / GQA) — 单 op, regime 在 forward 解析.

一个静态 `Attention.flash(...)` op 表示一层的 attention; `forward(step)` 解析为
prefill / decode / mixed:
  - 纯 prefill / 纯 decode: 现有 shape (num_tokens/num_seqs/q_len/kv_len), signature
    与 collector case 口径一致 (DB key 兼容)。
  - mixed (同时有 prefill+decode token, vLLM unified varlen kernel): 单个
    `op_subtype="mixed"`, shape={prefill_chunk/n_prefill/kv_prefill/n_decode/kv_decode}。
`roofline_spec` 按 regime 重算; mixed = prefill 段 + decode 段 spec 合成 (一个 fused
kernel 的总功做一次 roofline, 即 max(Σcompute, Σmem), 不是两次 roofline 之和)。

MLA (DeepSeek dense + sparse) 在 operators/mla.py 的 MLAAttention。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.graph.runtime import OpRuntime, StepRuntime
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import OperatorBase, RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


# ---- FlashAttention roofline formulas (pure; shape-driven) ----

def _flash_prefill_spec(
    *, seqlen: int, bs: int, n_q: int, n_kv: int, head_dim: int,
    a_byte: float, kv_byte: float, onchip: float,
) -> RooflineSpec:
    causal_factor = (seqlen + 1) / (2 * seqlen)
    flops = (
        seqlen * seqlen * head_dim * n_q * bs * 2          # Q @ K^T
        + seqlen * head_dim * seqlen * n_q * bs * 2        # softmax @ V
        + bs * n_q * seqlen * seqlen * 5                   # softmax
    ) * causal_factor
    block_size_r = min(math.ceil(onchip / (4 * kv_byte * head_dim)), head_dim)
    n_blocks_r = math.ceil(seqlen / block_size_r)
    q_numel = seqlen * head_dim * bs * n_q * a_byte
    o_numel = seqlen * head_dim * bs * n_q * a_byte
    kv_causal_factor = (n_blocks_r + 1) / (2 * n_blocks_r)
    kv_io = int(n_blocks_r * seqlen * head_dim * bs * n_kv * kv_byte * 2
                * kv_causal_factor)
    return RooflineSpec(
        op_category="attention",
        flops=int(flops),
        load_act=int(q_numel),
        store_act=int(o_numel * 2),
        load_kv_cache=kv_io,
    )


def _flash_decode_spec(
    *, ctx_len: int, bs: int, n_q: int, n_kv: int, head_dim: int,
    a_byte: float, kv_byte: float, onchip: float,
) -> RooflineSpec:
    flops = (
        ctx_len * head_dim * n_q * bs * 2
        + head_dim * ctx_len * n_q * bs * 2
        + bs * n_q * ctx_len * 5
    )
    block_size_r = min(math.ceil(onchip / (4 * kv_byte * head_dim)), head_dim)
    n_blocks_r = math.ceil(1 / block_size_r)
    q_numel = head_dim * bs * n_q * a_byte
    o_numel = head_dim * bs * n_q * a_byte
    kv_io = int(n_blocks_r * ctx_len * head_dim * bs * n_kv * kv_byte * 2)
    return RooflineSpec(
        op_category="attention",
        flops=int(flops),
        load_act=int(q_numel),
        store_act=int(o_numel * 2),
        load_kv_cache=kv_io,
    )


def _add_specs(a: RooflineSpec, b: RooflineSpec) -> RooflineSpec:
    """Sum two attention RooflineSpecs — mixed = prefill 段 + decode 段 的总功
    (一个 fused kernel; 由 analyzer 做一次 roofline = max(Σcompute, Σmem))."""
    return RooflineSpec(
        op_category="attention",
        flops=a.flops + b.flops,
        load_act=a.load_act + b.load_act,
        store_act=a.store_act + b.store_act,
        load_kv_cache=a.load_kv_cache + b.load_kv_cache,
    )


# signature shape keys: 标准 (prefill/decode) + mixed. 纯 prefill/decode 的 shape 不含
# mixed key → project 填 None → to_canonical 跳过 → hash 与旧版完全兼容。
_SHAPE_STD = ("num_tokens", "num_seqs", "q_len", "kv_len")
_SHAPE_MIXED = ("prefill_chunk", "n_prefill", "kv_prefill", "n_decode", "kv_decode")
_SHAPE_HEADS = ("num_q_heads", "num_kv_heads", "head_dim")
_SIG_SHAPE_KEYS = _SHAPE_STD + _SHAPE_MIXED + _SHAPE_HEADS


@dataclass(frozen=True)
class Attention(OperatorBase):
    """FlashAttention / GQA op (single, regime resolved at forward).

    构建期只持静态 head 维度 + backend/runtime; forward(step) 给出 prefill / decode /
    mixed 的 OpRuntime, roofline_spec 按 regime 重算。
    """
    name: str
    phase: str
    layer_idx: int | None
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    op_subtype: str = "attention"   # static placeholder; forward 给 prefill/decode/mixed
    attention_backend: str = "flash_attn"
    kv_dtype: str = "bf16"
    block_size: int = 0
    kernel_source: str = "vllm_flash_attn"
    dtype_override: str | None = None
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    count: int = 1
    # op_runtime=None (legacy/DB fallback) 用的 baked spec + 静态 shape; build-once 下
    # router 永远给 op_runtime → 走 forward 重算, 这些只是兜底。
    roofline_spec_value: RooflineSpec | None = None
    num_tokens: int = 0
    num_seqs: int = 0
    q_len: int = 0
    kv_len: int = 0

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
        """解析 regime: 纯 prefill / 纯 decode / mixed。空步 (无 prefill/decode token)
        → None (router 跳过, 无 attention work)。"""
        has_pf = step.num_prefill_tokens > 0
        has_dc = step.num_decode_requests > 0
        if not has_pf and not has_dc:
            return None
        heads = {"num_q_heads": self.num_q_heads, "num_kv_heads": self.num_kv_heads,
                 "head_dim": self.head_dim}
        if has_pf and has_dc:
            pf_bs = max(step.num_prefill_requests, 1)
            seqlen = step.max_prefill_seqlen
            shape = {
                "prefill_chunk": seqlen, "n_prefill": pf_bs,
                "kv_prefill": step.max_context_len if step.max_context_len > 0 else seqlen,
                "n_decode": step.num_decode_requests,
                "kv_decode": step.avg_decode_context_len,
                **heads,
            }
            subtype = "mixed"
        elif has_dc:
            bs = step.num_decode_requests
            shape = {"num_tokens": bs, "num_seqs": bs, "q_len": 1,
                     "kv_len": step.avg_decode_context_len, **heads}
            subtype = "decode"
        else:
            bs = max(step.num_prefill_requests, 1)
            seqlen = step.max_prefill_seqlen
            shape = {"num_tokens": bs * seqlen, "num_seqs": bs, "q_len": seqlen,
                     "kv_len": seqlen, **heads}
            subtype = "prefill"
        return OpRuntime(phase=step.phase, op_subtype=subtype, shape=shape,
                         parallel=dict(self.parallel), runtime=dict(self.runtime))

    def roofline_spec(self, op_runtime: OpRuntime | None = None) -> RooflineSpec:
        if op_runtime is None:
            return self.roofline_spec_value
        s = op_runtime.shape
        sub = op_runtime.op_subtype
        ab, kb, oc = self.ctx.a_byte, self.ctx.kv_byte, self.ctx.hw.onchip_buffer
        nq, nkv, hd = self.num_q_heads, self.num_kv_heads, self.head_dim
        if sub == "decode":
            return _flash_decode_spec(
                ctx_len=int(s["kv_len"]), bs=int(s["num_seqs"]),
                n_q=nq, n_kv=nkv, head_dim=hd, a_byte=ab, kv_byte=kb, onchip=oc)
        if sub == "mixed":
            pf = _flash_prefill_spec(
                seqlen=int(s["prefill_chunk"]), bs=int(s["n_prefill"]),
                n_q=nq, n_kv=nkv, head_dim=hd, a_byte=ab, kv_byte=kb, onchip=oc)
            dc = _flash_decode_spec(
                ctx_len=int(s["kv_decode"]), bs=int(s["n_decode"]),
                n_q=nq, n_kv=nkv, head_dim=hd, a_byte=ab, kv_byte=kb, onchip=oc)
            return _add_specs(pf, dc)
        return _flash_prefill_spec(
            seqlen=int(s["q_len"]), bs=int(s["num_seqs"]),
            n_q=nq, n_kv=nkv, head_dim=hd, a_byte=ab, kv_byte=kb, onchip=oc)

    def signature(self, op_runtime: OpRuntime | None = None) -> OperatorSignature:
        return self.resolved_signature(
            op_runtime,
            shape_keys=_SIG_SHAPE_KEYS,
            parallel_keys=("tp",),
            runtime_keys=(
                "framework", "framework_version", "execution_mode",
                "kernel_source", "attention_backend", "kv_dtype", "block_size",
            ),
        )

    @classmethod
    def flash(
        cls, *,
        layer_idx: int, n_q: int, n_kv: int, head_dim: int,
        ctx: OperatorContext,
        name: str = "attention",
        count: int = 1,
        phase: str = "",
        kernel_source: str = "vllm_flash_attn",
        tags: tuple[str, ...] = (),
    ) -> "Attention":
        """单 flash/GQA attention op. forward(step) 解析 prefill/decode/mixed.

        n_q / n_kv 已 / TP. baked fallback (op_runtime=None) 用 seqlen=1 占位 prefill
        spec; 实际 cost 由 forward + roofline_spec 按 step 重算。
        """
        placeholder = _flash_prefill_spec(
            seqlen=1, bs=1, n_q=n_q, n_kv=n_kv, head_dim=head_dim,
            a_byte=ctx.a_byte, kv_byte=ctx.kv_byte, onchip=ctx.hw.onchip_buffer,
        )
        return cls(
            name=name, phase=phase, layer_idx=layer_idx,
            num_q_heads=n_q, num_kv_heads=n_kv, head_dim=head_dim,
            attention_backend="flash_attn", kv_dtype="bf16",
            block_size=ctx.block_size, kernel_source=kernel_source,
            ctx=ctx, tags=tags, count=count,
            roofline_spec_value=placeholder,
        )
