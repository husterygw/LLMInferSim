"""Attention operators: standard (qk/sv/softmax) and FlashAttention-2 fused.

Formulas are extracted verbatim from model_analyzer.py to ensure identical output.

阶段 3 新增:
  - rope_kernel : 独立 in-place bandwidth-bound kernel (详设 §4.7.1a (5))
"""

import math
from llm_infer_sim.core.ops.base import OperatorProfile


def rope_kernel(
    name: str,
    tokens: int,
    num_q_heads_per_tp: int,
    num_kv_heads_per_tp: int,
    head_dim: int,
    a_byte: float,
) -> OperatorProfile:
    """RoPE 独立 in-place kernel (详设 §4.7.1a (5))。

    vLLM 的 RoPE 不与 attention 融合, 单独 op:
      - 修改 Q [tokens, num_q_heads*head_dim] in-place
      - 修改 K [tokens, num_kv_heads*head_dim] in-place
      - bandwidth-bound, FLOPs 微小 (cos/sin lookup + 2 mul + 1 add 每对元素)
    """
    qk_elements = tokens * (num_q_heads_per_tp + num_kv_heads_per_tp) * head_dim
    from llm_infer_sim.core.profiles.shape_buckets import (
        OP_KIND_ROPE, dense_efficiency_key,
    )
    # Per element FLOPs: 3 (RoPE rotates 2 元素一对, 每对 4 mul + 2 add = 6 ops,
    # 摊到每元素 = 3 ops). 历史: 旧值 6 把 "per pair" 当 "per element" 算, double-count.
    return OperatorProfile(
        name=name,
        op_category="activation",
        flops=qk_elements * 3,
        load_act=int(qk_elements * a_byte),              # in-place: 读 Q+K
        store_act=int(qk_elements * a_byte),             # in-place: 写 Q+K
        efficiency_key=dense_efficiency_key(OP_KIND_ROPE, tokens),
    )


def attention_decode_standard(
    seqlen: int,
    batchsize: int,
    num_attention_heads: int,
    num_key_value_heads: float,
    head_size: int,
    a_byte: float,
    kv_byte: float,
) -> list[OperatorProfile]:
    """Standard (non-flash) decode attention for MHA/GQA: [qk_matmul, sv_matmul, softmax].

    MLA decode 用专用 `attention_decode_mla` (FlashMLA kernel, kv-absorbed),
    不走本函数。
    """
    _qk = head_size
    _sv = head_size

    qk_ops = seqlen * _qk * num_attention_heads * batchsize * 2
    sv_ops = 1 * _sv * seqlen * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * seqlen * 1 * 5

    qk_kv_io = int(seqlen * head_size * batchsize * num_key_value_heads * kv_byte)
    sv_kv_io = int(seqlen * head_size * batchsize * num_key_value_heads * kv_byte)
    q_io = int(1 * head_size * batchsize * num_attention_heads * a_byte)
    o_io = int(1 * head_size * batchsize * num_attention_heads * a_byte)

    qk = OperatorProfile(
        name="qk_matmul",
        op_category="attention",
        flops=qk_ops,
        load_act=q_io,
        store_act=o_io,
        load_kv_cache=qk_kv_io,
    )
    sv = OperatorProfile(
        name="sv_matmul",
        op_category="attention",
        flops=sv_ops,
        load_act=int(1 * seqlen * batchsize * num_attention_heads * a_byte),
        store_act=o_io,
        load_kv_cache=sv_kv_io,
    )
    soft = OperatorProfile(
        name="softmax",
        op_category="attention",
        flops=softmax_ops,
        load_act=int(batchsize * num_attention_heads * seqlen * 1 * a_byte),
        store_act=int(batchsize * num_attention_heads * seqlen * 1 * a_byte),
    )
    return [qk, sv, soft]


def attention_decode_flash(
    seqlen: int,
    batchsize: int,
    num_attention_heads: int,
    num_key_value_heads: float,
    head_size: int,
    a_byte: float,
    kv_byte: float,
    onchip_buffer: float,
) -> list[OperatorProfile]:
    """FlashAttention-2 fused decode attention for MHA/GQA.

    MLA decode 用专用 `attention_decode_mla` (FlashMLA kernel, kv-absorbed),
    不走本函数。
    """
    _qk = head_size
    _sv = head_size

    qk_ops = seqlen * _qk * num_attention_heads * batchsize * 2
    sv_ops = 1 * _sv * seqlen * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * seqlen * 1 * 5

    block_size_r = min(math.ceil(onchip_buffer / (kv_byte * _qk)), _qk)
    n_blocks_r = math.ceil(1 / block_size_r)
    q_numel = 1 * _qk * batchsize * num_attention_heads * a_byte
    o_numel = 1 * _sv * batchsize * num_attention_heads * a_byte

    kv_io = int(n_blocks_r * seqlen * head_size * batchsize * num_key_value_heads * kv_byte * 2)

    fused = OperatorProfile(
        name="fused_attention",
        op_category="attention",
        flops=qk_ops + sv_ops + softmax_ops,
        load_act=int(q_numel),
        store_act=int(o_numel * 2),
        load_kv_cache=kv_io,
    )
    return [fused]


def attention_decode_mla(
    ctx_len: int,
    batchsize: int,
    num_attention_heads: int,
    kv_latent_dim: int,
    kv_lora_rank: int,
    a_byte: float,
    kv_byte: float,
    onchip_buffer: float,
) -> list[OperatorProfile]:
    """FlashMLA fused decode attention (DeepSeek-V2/V3, kv-absorbed).

    vLLM 真实路径 (`/vllm/v1/attention/ops/flashmla.py` 调 FlashMLA kernel):
      - KV cache 存 c_kv (kv_latent_dim = kv_lora_rank + qk_rope_head_dim, single-head)
      - Q 投到 absorbed 维 (kv_latent_dim), 跟 cache 算 dot → softmax →
        scores × c_kv (kv_lora_rank) → output (per-head v dim)
      - 跟 MHA/GQA FlashAttention-2 的差异:
          * QK matmul: Q[heads, kv_latent_dim] × K[seqlen, kv_latent_dim], 不切 head_dim
          * SV matmul: S[heads, seqlen] × V[seqlen, kv_lora_rank], 共享 V (single head)
          * KV cache IO = seqlen × kv_latent_dim × kv_byte (NO num_kv_heads 因子)
    """
    _qk = kv_latent_dim                  # absorbed Q dim for QK
    _sv = kv_lora_rank or kv_latent_dim  # c_kv dim for SV output

    qk_ops = ctx_len * _qk * num_attention_heads * batchsize * 2
    sv_ops = 1 * _sv * ctx_len * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * ctx_len * 1 * 5

    block_size_r = min(math.ceil(onchip_buffer / (kv_byte * _qk)), _qk)
    n_blocks_r = math.ceil(1 / block_size_r)
    q_numel = 1 * _qk * batchsize * num_attention_heads * a_byte
    o_numel = 1 * _sv * batchsize * num_attention_heads * a_byte

    # MLA cache: single-head c_kv per position, dim = kv_latent_dim.
    kv_io = int(n_blocks_r * ctx_len * kv_latent_dim * batchsize * kv_byte)

    fused = OperatorProfile(
        name="fused_mla_attention",
        op_category="attention",
        flops=qk_ops + sv_ops + softmax_ops,
        load_act=int(q_numel),
        store_act=int(o_numel * 2),
        load_kv_cache=kv_io,
    )
    return [fused]


def attention_prefill_standard(
    seqlen: int,
    batchsize: int,
    num_attention_heads: int,
    num_key_value_heads: float,
    head_size: int,
    a_byte: float,
    kv_byte: float,
) -> list[OperatorProfile]:
    """Standard (non-flash) prefill attention for MHA/GQA: [qk_matmul, sv_matmul, softmax].

    MLA prefill 用专用 `attention_prefill_mla`, 不走本函数。
    """
    _qk = head_size
    _v = head_size

    qk_ops = seqlen * seqlen * _qk * num_attention_heads * batchsize * 2
    sv_ops = seqlen * _v * seqlen * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * seqlen * seqlen * 5

    qk = OperatorProfile(
        name="qk_matmul",
        op_category="attention",
        flops=qk_ops,
        load_act=int(seqlen * _qk * batchsize * num_attention_heads * a_byte),
        store_act=int(seqlen * _qk * batchsize * num_attention_heads * a_byte),
        load_kv_cache=int(seqlen * _qk * batchsize * num_key_value_heads * kv_byte),
    )
    sv = OperatorProfile(
        name="sv_matmul",
        op_category="attention",
        flops=sv_ops,
        load_act=int(seqlen * seqlen * batchsize * num_attention_heads * a_byte),
        store_act=int(seqlen * _v * batchsize * num_attention_heads * a_byte),
        load_kv_cache=int(seqlen * _v * batchsize * num_key_value_heads * kv_byte),
    )
    soft = OperatorProfile(
        name="softmax",
        op_category="attention",
        flops=softmax_ops,
        load_act=int(batchsize * num_attention_heads * seqlen * seqlen * a_byte),
        store_act=int(batchsize * num_attention_heads * seqlen * seqlen * a_byte),
    )
    return [qk, sv, soft]


def attention_prefill_flash(
    seqlen: int,
    batchsize: int,
    num_attention_heads: int,
    num_key_value_heads: float,
    head_size: int,
    a_byte: float,
    kv_byte: float,
    onchip_buffer: float,
) -> list[OperatorProfile]:
    """FlashAttention-2 fused prefill attention for MHA/GQA.

    MLA prefill 用专用 `attention_prefill_mla`, 不走本函数。
    """
    _qk = head_size
    _v = head_size

    qk_ops = seqlen * seqlen * _qk * num_attention_heads * batchsize * 2
    sv_ops = seqlen * _v * seqlen * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * seqlen * seqlen * 5

    block_size_r = min(math.ceil(onchip_buffer / (kv_byte * _qk)), _qk)
    n_blocks_r = math.ceil(seqlen / block_size_r)
    q_numel = seqlen * _qk * batchsize * num_attention_heads * a_byte
    o_numel = seqlen * _v * batchsize * num_attention_heads * a_byte

    kv_io = int(n_blocks_r * seqlen * head_size * batchsize * num_key_value_heads * kv_byte * 2)

    fused = OperatorProfile(
        name="fused_attention",
        op_category="attention",
        flops=qk_ops + sv_ops + softmax_ops,
        load_act=int(q_numel),
        store_act=int(o_numel * 2),
        load_kv_cache=kv_io,
    )
    return [fused]


def attention_decode_mla_sparse(
    ctx_len: int,
    batchsize: int,
    num_attention_heads: int,
    kv_latent_dim: int,
    kv_lora_rank: int,
    index_topk: int,
    a_byte: float,
    kv_byte: float,
    onchip_buffer: float,
) -> list[OperatorProfile]:
    """DSA decode attention (DeepSeek-V3.2-Exp, sparse MLA + FlashMLA).

    跟 `attention_decode_mla` 的区别: attended_len = min(ctx_len, index_topk) 而非 ctx_len.
    Indexer 选出 top-k positions, 主 attention 只在这 top-k 上做 (vLLM FlashMLASparse).
    其余公式 (kv-absorbed Q[kv_latent_dim] × K[kv_latent_dim], SV with c_kv) 跟 MLA 一致.
    """
    attended = min(ctx_len, index_topk) if index_topk > 0 else ctx_len
    _qk = kv_latent_dim
    _sv = kv_lora_rank or kv_latent_dim

    qk_ops = attended * _qk * num_attention_heads * batchsize * 2
    sv_ops = 1 * _sv * attended * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * attended * 1 * 5

    block_size_r = min(math.ceil(onchip_buffer / (kv_byte * _qk)), _qk)
    n_blocks_r = math.ceil(1 / block_size_r)
    q_numel = 1 * _qk * batchsize * num_attention_heads * a_byte
    o_numel = 1 * _sv * batchsize * num_attention_heads * a_byte
    # Cache 读量按 sparse attended (top-k 选中的 c_kv positions).
    kv_io = int(n_blocks_r * attended * kv_latent_dim * batchsize * kv_byte)

    fused = OperatorProfile(
        name="fused_mla_sparse_attention",
        op_category="attention",
        flops=qk_ops + sv_ops + softmax_ops,
        load_act=int(q_numel),
        store_act=int(o_numel * 2),
        load_kv_cache=kv_io,
    )
    return [fused]


def attention_prefill_mla_sparse(
    seqlen: int,
    batchsize: int,
    num_attention_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_latent_dim: int,
    index_topk: int,
    a_byte: float,
    kv_byte: float,
    onchip_buffer: float,
    prior_ctx_tokens: int = 0,
) -> list[OperatorProfile]:
    """DSA prefill attention (DeepSeek-V3.2-Exp, sparse MLA).

    跟 `attention_prefill_mla` 的区别: per-query attended 上界 = index_topk (而非 seqlen+prior_ctx).
    总 attended = sum over query pos of min(pos+1+prior_ctx, index_topk).
    简化: 用 avg attended ≈ min((seqlen+1)/2 + prior_ctx, index_topk) 作为 per-pos 平均.
    总 attended_sum = seqlen × avg_attended.
    """
    avg_ctx = (seqlen + 1) / 2 + prior_ctx_tokens
    avg_attended = min(avg_ctx, float(index_topk)) if index_topk > 0 else avg_ctx
    attended_sum = int(seqlen * avg_attended)

    qk_ops = int(attended_sum * qk_head_dim * num_attention_heads * batchsize * 2)
    sv_ops = int(attended_sum * v_head_dim * num_attention_heads * batchsize * 2)
    softmax_ops = int(batchsize * num_attention_heads * attended_sum * 5)

    block_size_r = min(math.ceil(onchip_buffer / (kv_byte * qk_head_dim)), qk_head_dim)
    n_blocks_r = math.ceil(seqlen / block_size_r)
    q_numel = seqlen * qk_head_dim * batchsize * num_attention_heads * a_byte
    o_numel = seqlen * v_head_dim * batchsize * num_attention_heads * a_byte
    # K/V staging activation (sparse: 只读 top-k 选中的 c_kv 重算 K/V, 不是全 ctx).
    # 跟 dense MLA 一样: K/V 是 kv_b_proj 输出, MHA 形状 × avg_attended.
    kv_staging_act = int(
        n_blocks_r * avg_attended * (qk_head_dim + v_head_dim) * batchsize * num_attention_heads * a_byte
    )
    # Prior ctx cache 读 (c_kv single-head), 同样受 index_topk 限制.
    if prior_ctx_tokens > 0:
        prior_attended = min(prior_ctx_tokens, index_topk) if index_topk > 0 else prior_ctx_tokens
        kv_io = int(prior_attended * kv_latent_dim * batchsize * kv_byte)
    else:
        kv_io = 0

    fused = OperatorProfile(
        name="fused_mla_sparse_attention",
        op_category="attention",
        flops=qk_ops + sv_ops + softmax_ops,
        load_act=int(q_numel + kv_staging_act),
        store_act=int(o_numel * 2),
        load_kv_cache=kv_io,
    )
    return [fused]


def attention_prefill_mla(
    seqlen: int,
    batchsize: int,
    num_attention_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_latent_dim: int,
    a_byte: float,
    kv_byte: float,
    onchip_buffer: float,
    prior_ctx_tokens: int = 0,
) -> list[OperatorProfile]:
    """MLA prefill attention (DeepSeek-V2/V3, FlashAttn-2 with diff_headdims).

    vLLM 真实路径 (`mla_attention.py:2341-2355` `_run_prefill_new_tokens_fa` →
    `flash_attn_varlen_diff_headdims`):
      - 新 token prefill: K/V 来自上游 `kv_b_proj` staging activation (MHA 形状
        [seqlen, num_heads, qk_head_dim] + [seqlen, num_heads, v_head_dim]),
        不读 KV cache; activation IO 用 a_byte (非 kv_byte).
      - Chunked prefill 跨 step (`mla_attention.py:2596-2650`): prior context
        从 paged cache gather c_kv (kv_latent_dim × 1 head), 再 kv_b_proj 重算.
        cache 读量是 c_kv 维 (single-head), 不是 decompressed K+V × num_heads.
      - QK / SV 用 FlashAttention-2 with different headdims (qk_dim ≠ v_dim,
        V3 是 192 vs 128); kernel 跑 MHA 形状 (num_heads × qk_dim).
    """
    qk_ops = seqlen * seqlen * qk_head_dim * num_attention_heads * batchsize * 2
    sv_ops = seqlen * v_head_dim * seqlen * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * seqlen * seqlen * 5

    block_size_r = min(math.ceil(onchip_buffer / (kv_byte * qk_head_dim)), qk_head_dim)
    n_blocks_r = math.ceil(seqlen / block_size_r)
    q_numel = seqlen * qk_head_dim * batchsize * num_attention_heads * a_byte
    o_numel = seqlen * v_head_dim * batchsize * num_attention_heads * a_byte
    # K/V staging activation (kv_b_proj 输出 + FlashAttn 重读 n_blocks_r 次), 走 load_act.
    kv_staging_act = int(
        n_blocks_r * seqlen * (qk_head_dim + v_head_dim) * batchsize * num_attention_heads * a_byte
    )
    # Chunked prefill prior context: 从 cache 读 c_kv (single-head latent).
    if prior_ctx_tokens > 0:
        kv_io = int(prior_ctx_tokens * kv_latent_dim * batchsize * kv_byte)
    else:
        kv_io = 0

    fused = OperatorProfile(
        name="fused_mla_attention",
        op_category="attention",
        flops=qk_ops + sv_ops + softmax_ops,
        load_act=int(q_numel + kv_staging_act),
        store_act=int(o_numel * 2),
        load_kv_cache=kv_io,
    )
    return [fused]


# ---------------------------------------------------------------------------
# V4 sparse attention (sliding window + compressed + indexed positions)
# ---------------------------------------------------------------------------

def attention_prefill_sparse(
    seqlen: int,
    batchsize: int,
    num_attention_heads: int,
    head_size: int,
    a_byte: float,
    kv_byte: float,
    window_size: int,
    compress_ratio: int,
    index_topk: int = 0,
    onchip_buffer: float | None = None,
) -> list[OperatorProfile]:
    """V4 sparse prefill attention (fused).

    Each causal query position attends to:
    - local window: min(pos + 1, window_size)
    - compressed cache: (pos + 1) // compress_ratio
    - indexed compressed positions: min(index_topk, compressed_attended)

    KV is single-head (shared across all query heads): 1 KV head, head_dim per position.
    """
    total_attended = 0
    total_kv_positions_loaded = 0
    if onchip_buffer and onchip_buffer > 0:
        block_size_r = max(1, math.floor(onchip_buffer / (kv_byte * head_size)))
    else:
        block_size_r = None
    for pos in range(seqlen):
        local_attended = min(pos + 1, window_size)
        all_compressed = (pos + 1) // compress_ratio if compress_ratio > 0 else 0
        if compress_ratio > 0 and index_topk > 0:
            # CSA: indexer selects top-k from compressed positions (replaces all-compressed)
            compressed_attended = min(index_topk, all_compressed)
        else:
            # HCA or no compression: attend to ALL compressed positions
            compressed_attended = all_compressed
        # 阶段 9 fix 17: attn_sink softmax 包含 1 个 -inf init sink token
        attn_sink = 1
        attended = local_attended + compressed_attended + attn_sink
        total_attended += attended
        if block_size_r is None:
            total_kv_positions_loaded += attended
        else:
            n_blocks_r = math.ceil(attended / block_size_r)
            total_kv_positions_loaded += n_blocks_r * attended

    qk_ops = total_attended * head_size * num_attention_heads * batchsize * 2
    sv_ops = total_attended * head_size * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * total_attended * 5

    q_numel = seqlen * head_size * batchsize * num_attention_heads * a_byte
    o_numel = seqlen * head_size * batchsize * num_attention_heads * a_byte

    # KV cache IO: single-head KV, optionally reloaded in chunks if on-chip buffer
    # cannot hold the whole attended context.
    kv_io = int(total_kv_positions_loaded * head_size * batchsize * kv_byte)

    fused = OperatorProfile(
        name="fused_sparse_attention",
        op_category="attention",
        flops=qk_ops + sv_ops + softmax_ops,
        load_act=int(q_numel),
        store_act=int(o_numel * 2),
        load_kv_cache=kv_io,
    )
    return [fused]


def attention_decode_sparse(
    ctx_len: int,
    batchsize: int,
    num_attention_heads: int,
    head_size: int,
    a_byte: float,
    kv_byte: float,
    window_size: int,
    compress_ratio: int,
    index_topk: int = 0,
    onchip_buffer: float | None = None,
) -> list[OperatorProfile]:
    """V4 sparse decode attention (fused).

    Single new token attends to: window_size + ctx_len/ratio + index_topk positions.
    KV is single-head: 1 KV head with head_dim per position.
    """
    all_compressed = ctx_len // compress_ratio if compress_ratio > 0 else 0
    if compress_ratio > 0 and index_topk > 0:
        # CSA: indexer selects top-k from compressed positions (replaces all-compressed)
        compressed_attended = min(index_topk, all_compressed)
    else:
        # HCA or no compression: attend to ALL compressed positions
        compressed_attended = all_compressed
    # 阶段 9 fix 17: attn_sink +1 attended pos (softmax 包含 1 个 sink)
    attn_sink = 1
    attended = min(ctx_len, window_size) + compressed_attended + attn_sink

    qk_ops = attended * head_size * num_attention_heads * batchsize * 2
    sv_ops = 1 * head_size * attended * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * attended * 1 * 5

    q_io = int(1 * head_size * batchsize * num_attention_heads * a_byte)
    o_io = int(1 * head_size * batchsize * num_attention_heads * a_byte)
    # KV cache: load attended positions × head_size (single head, no num_kv_heads factor)
    if onchip_buffer and onchip_buffer > 0:
        block_size_r = max(1, math.floor(onchip_buffer / (kv_byte * head_size)))
        n_blocks_r = math.ceil(attended / block_size_r)
    else:
        n_blocks_r = 1
    kv_io = int(n_blocks_r * attended * head_size * batchsize * kv_byte)

    fused = OperatorProfile(
        name="fused_sparse_attention",
        op_category="attention",
        flops=qk_ops + sv_ops + softmax_ops,
        load_act=q_io,
        store_act=int(o_io * 2),
        load_kv_cache=kv_io,
    )
    return [fused]
