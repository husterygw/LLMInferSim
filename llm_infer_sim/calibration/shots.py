"""Shot 网格 — Layer 1 microbench 要测的合成 batch shape 列表 (详设 §9.4.2 B.1).

一个 Shot 表示一个合成 SchedulerOutput 的 shape, 用于驱动 vLLM
model_runner.execute_model 跑出某种调度形态:

  - dense: 纯 prefill, num_decode=0, 不同 token 数; 校 GEMM / norm / rope / etc.
  - attention: 跨 attention kernel 多种 (prefill_chunk, kv_lens, n_decode) 组合
  - per_sequence: 1-token-per-seq decode, 不同 batch_size; 校 sampler / lm_head 等
                  shape 跟 num_seqs 强相关 (而不是 num_tokens) 的 op

跨进程序列化: 用 to_dict / hydrate 在 worker_extension 跟主进程间传递。

设计参考 LLMServingSim profiler/core/hooks/batch.py 的 Shot 概念, 简化为单结构体
(不分 Shot/PerSeqShot/AttnShot)。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Shot:
    """合成 batch 的 shape 描述, 用作 profile 一次测量的 key.

    字段语义:
        kind: "dense" | "attention" | "per_sequence" — 决定 shot 该归入哪个 CSV
        num_new_tokens: 本 step 调度的 prefill / chunked-prefill token 总数
                        (decode shot 该字段 = num_decode_seqs, 因为每 decode seq 1 token)
        num_decode_seqs: 本 step 同时跑 decode 的 sequence 数
        kv_lens_prefill: per prefill-seq 的 ctx_len (chunked prefill 续段时 > 0)
        kv_lens_decode: per decode-seq 的 ctx_len
        prefill_chunk: 仅 attention shot 用 — 此次 chunked prefill 已 forward 多少 token

    shot key (CSV 行键): 跟 kind 相关
        dense:        (num_new_tokens,)               → CSV "layer,tokens,time_us"
        attention:    (prefill_chunk, kv_prefill, n_decode, kv_decode)
        per_sequence: (num_decode_seqs,)              → CSV "layer,sequences,time_us"
    """
    kind: str
    num_new_tokens: int = 0
    num_decode_seqs: int = 0
    kv_lens_prefill: list[int] = field(default_factory=list)
    kv_lens_decode: list[int] = field(default_factory=list)
    prefill_chunk: int = 0     # 仅 attention shot

    # ---- 序列化 (跨进程传 worker_extension.fire) ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "num_new_tokens": self.num_new_tokens,
            "num_decode_seqs": self.num_decode_seqs,
            "kv_lens_prefill": list(self.kv_lens_prefill),
            "kv_lens_decode": list(self.kv_lens_decode),
            "prefill_chunk": self.prefill_chunk,
        }

    @classmethod
    def hydrate(cls, d: dict[str, Any]) -> "Shot":
        return cls(
            kind=str(d["kind"]),
            num_new_tokens=int(d.get("num_new_tokens", 0)),
            num_decode_seqs=int(d.get("num_decode_seqs", 0)),
            kv_lens_prefill=list(d.get("kv_lens_prefill", [])),
            kv_lens_decode=list(d.get("kv_lens_decode", [])),
            prefill_chunk=int(d.get("prefill_chunk", 0)),
        )

    # ---- CSV 列值 ----

    def csv_key(self) -> tuple:
        """返回该 shot 在 CSV 里作 row 标识的列值组合."""
        if self.kind == "dense":
            return (self.num_new_tokens,)
        if self.kind == "attention":
            kv_pref = self.kv_lens_prefill[0] if self.kv_lens_prefill else 0
            kv_dec = self.kv_lens_decode[0] if self.kv_lens_decode else 0
            return (self.prefill_chunk, kv_pref, self.num_decode_seqs, kv_dec)
        if self.kind == "per_sequence":
            return (self.num_decode_seqs,)
        raise ValueError(f"Unknown shot kind: {self.kind}")


# ---------------------------------------------------------------------------
# 预设网格 (B.1: 覆盖 4090 + Qwen3-4B 主要场景; 后续 phase 按需扩)
# ---------------------------------------------------------------------------

# Dense: 纯 prefill, 测 GEMM / norm / rope / SwiGLU. token 对数桶 + 几个标定点.
_DENSE_TOKEN_GRID = (1, 8, 64, 128, 256, 512, 1024, 2048)
DENSE_SHOTS: tuple[Shot, ...] = tuple(
    Shot(kind="dense", num_new_tokens=t) for t in _DENSE_TOKEN_GRID
)

# Per-sequence: 1-tok-per-seq decode 同 batch 但 sequence 数不同. 测 sampler / lm_head.
# 实测 lm_head time 跟 #sequences 强相关 (decode 一次输出 #seq tokens).
_PER_SEQ_GRID = (1, 2, 4, 8, 16, 32, 64)
PER_SEQUENCE_SHOTS: tuple[Shot, ...] = tuple(
    Shot(kind="per_sequence",
         num_new_tokens=n, num_decode_seqs=n,
         kv_lens_decode=[256] * n)        # 固定 ctx 256, 隔离 sequence 维度
    for n in _PER_SEQ_GRID
)

# Attention: 4D 网格 (prefill_chunk, kv_prefill, n_decode, kv_decode).
# 4090 24GB - 7.6GB weights - buffer ≈ 16GB → KV ~100k tokens at fp16.
# 网格按 n_decode × kv_decode 总 token 数 ≤ ~16k 装得下 single shot (max_model_len=20480 限制).
# 16k = max_model_len 上限, 单 seq 16k 是极限点.
_ATTN_DECODE_POINTS = [
    # n_decode=1: 单 seq, kv 可大
    (1, 256), (1, 1024), (1, 4096), (1, 16384),
    # n_decode=4: 每 seq kv ≤ 4096 → 总 16k
    (4, 256), (4, 1024), (4, 4096),
    # n_decode=16: 每 seq kv ≤ 1024 → 总 16k
    (16, 256), (16, 1024),
]
_ATTN_PREFILL_POINTS = list(itertools.product(
    [128, 512, 2048],      # prefill_chunk
    [0],                   # kv_prefill (从 0 开始的 prefill)
))
_ATTN_MIXED_POINTS = [
    (128, 4, 2048),
    (512, 4, 2048),
]

ATTENTION_SHOTS: tuple[Shot, ...] = tuple(
    # decode-only
    [
        Shot(kind="attention",
             num_new_tokens=n_dec, num_decode_seqs=n_dec,
             kv_lens_decode=[kv_dec] * n_dec)
        for n_dec, kv_dec in _ATTN_DECODE_POINTS
    ]
    # prefill-only (chunk size 决定本 step 调度的 token)
    + [
        Shot(kind="attention",
             num_new_tokens=chunk, prefill_chunk=chunk,
             kv_lens_prefill=[kv_pre])
        for chunk, kv_pre in _ATTN_PREFILL_POINTS
    ]
    # mixed (chunked prefill + decode 同 step)
    + [
        Shot(kind="attention",
             num_new_tokens=chunk + n_dec,
             num_decode_seqs=n_dec, prefill_chunk=chunk,
             kv_lens_prefill=[0],
             kv_lens_decode=[kv_dec] * n_dec)
        for chunk, n_dec, kv_dec in _ATTN_MIXED_POINTS
    ]
)


def all_shots_for_kind(kind: str) -> tuple[Shot, ...]:
    if kind == "dense":
        return DENSE_SHOTS
    if kind == "per_sequence":
        return PER_SEQUENCE_SHOTS
    if kind == "attention":
        return ATTENTION_SHOTS
    raise ValueError(f"Unknown shot kind: {kind}")
