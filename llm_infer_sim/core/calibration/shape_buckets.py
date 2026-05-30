"""Shape bucket 函数 — efficiency_key 共享逻辑 (详设 §9.4.2 B.6).

op 构造函数 (linear_layer / norm_layer / ...) 和 fit.py 都用这套桶函数算 shape_key,
保证 calibration 落 YAML 跟 cost model lookup 用同一份字符串 key, 不漂移.

桶粒度选择: 4090 + Qwen3-4B 实测够分辨 4 档:
  - decode (token <= 16)
  - small batch (16 < tokens <= 128)
  - mid batch (128 < tokens <= 1024)
  - prefill (tokens > 1024)
后续如要细化推到 X.2 (multi-axis bucket).
"""
from __future__ import annotations


# ---- op_kind 常量 (跟 catalog YAML 对齐) ----

OP_KIND_DENSE_GEMM = "dense_gemm"
OP_KIND_RMSNORM = "rmsnorm"
OP_KIND_SWIGLU = "swiglu"
OP_KIND_ROPE = "rope"
OP_KIND_EMBEDDING = "embedding"
OP_KIND_ATTN = "attn"


# ---- token-axis 桶 (dense / per_layer ops) ----

def token_bucket(tokens: int) -> str:
    if tokens <= 16:
        return "tokens<=16"
    if tokens <= 128:
        return "tokens<=128"
    if tokens <= 1024:
        return "tokens<=1024"
    return "tokens>1024"


# ---- sequence-axis 桶 (per_sequence ops: lm_head / sampler) ----

def sequence_bucket(sequences: int) -> str:
    if sequences <= 4:
        return "seq<=4"
    if sequences <= 16:
        return "seq<=16"
    return "seq>16"


# ---- kv-axis 桶 (attention KV cache 维度) ----

def kv_bucket(kv_len: int) -> str:
    if kv_len <= 256:
        return "kv<=256"
    if kv_len <= 1024:
        return "kv<=1024"
    if kv_len <= 4096:
        return "kv<=4k"
    if kv_len <= 16384:
        return "kv<=16k"
    return "kv>16k"


# ---- attention 多维桶 ----

def attention_bucket(
    prefill_chunk: int,
    kv_prefill: int,
    n_decode: int,
    kv_decode: int,
) -> str:
    """attention shot 二维桶: regime | kv_bucket.

    regime ∈ {prefill_tokens<=*, decode}; kv 取 prefill/decode 中较大者.
    """
    if prefill_chunk > 0:
        regime = f"prefill_{token_bucket(prefill_chunk)}"
    else:
        regime = "decode"
    kv = max(kv_prefill, kv_decode)
    return f"{regime}|{kv_bucket(kv)}"


# ---- 工具: 取 efficiency_key (op_kind, shape_key) tuple ----

def dense_efficiency_key(op_kind: str, tokens: int) -> tuple[str, str]:
    return (op_kind, token_bucket(tokens))


def per_seq_efficiency_key(op_kind: str, sequences: int) -> tuple[str, str]:
    return (op_kind, sequence_bucket(sequences))


def attention_efficiency_key(
    prefill_chunk: int, kv_prefill: int, n_decode: int, kv_decode: int,
) -> tuple[str, str]:
    return (OP_KIND_ATTN,
            attention_bucket(prefill_chunk, kv_prefill, n_decode, kv_decode))
