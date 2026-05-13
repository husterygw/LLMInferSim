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
    return OperatorProfile(
        name=name,
        op_category="activation",
        flops=qk_elements * 6,                           # 4 mul + 2 add per pair
        load_act=int(qk_elements * a_byte),              # in-place: 读 Q+K
        store_act=int(qk_elements * a_byte),             # in-place: 写 Q+K
    )


def attention_decode_standard(
    seqlen: int,
    batchsize: int,
    num_attention_heads: int,
    num_key_value_heads: float,
    head_size: int,
    a_byte: float,
    kv_byte: float,
    kv_latent_dim: int | None = None,
    kv_lora_rank: int | None = None,
) -> list[OperatorProfile]:
    """Standard (non-flash) decode attention: returns [qk_matmul, sv_matmul, softmax]."""
    is_mla = kv_latent_dim is not None and kv_latent_dim > 0
    if is_mla:
        _qk = kv_latent_dim        # absorbed Q dim for QK
        _sv = kv_lora_rank or kv_latent_dim  # c_kv dim for SV
    else:
        _qk = head_size
        _sv = head_size

    qk_ops = seqlen * _qk * num_attention_heads * batchsize * 2
    sv_ops = 1 * _sv * seqlen * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * seqlen * 1 * 5

    if is_mla:
        # MLA decode: compressed KV cache is a single tensor (kv_latent_dim per position)
        qk_kv_io = int(seqlen * kv_latent_dim * batchsize * kv_byte)
        sv_kv_io = int(seqlen * _sv * batchsize * kv_byte)
        q_io = int(1 * _qk * batchsize * num_attention_heads * a_byte)
        o_io = int(1 * _sv * batchsize * num_attention_heads * a_byte)
    else:
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
    kv_latent_dim: int | None = None,
    kv_lora_rank: int | None = None,
) -> list[OperatorProfile]:
    """FlashAttention-2 fused decode attention."""
    is_mla = kv_latent_dim is not None and kv_latent_dim > 0
    if is_mla:
        _qk = kv_latent_dim        # absorbed Q dim for QK
        _sv = kv_lora_rank or kv_latent_dim  # c_kv dim for SV
    else:
        _qk = head_size
        _sv = head_size

    qk_ops = seqlen * _qk * num_attention_heads * batchsize * 2
    sv_ops = 1 * _sv * seqlen * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * seqlen * 1 * 5

    block_size_r = min(math.ceil(onchip_buffer / (kv_byte * _qk)), _qk)
    n_blocks_r = math.ceil(1 / block_size_r)
    q_numel = 1 * _qk * batchsize * num_attention_heads * a_byte
    o_numel = 1 * _sv * batchsize * num_attention_heads * a_byte

    if is_mla:
        # MLA decode: compressed KV cache is a single tensor (kv_latent_dim per position)
        kv_io = int(n_blocks_r * seqlen * kv_latent_dim * batchsize * kv_byte)
    else:
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


def attention_prefill_standard(
    seqlen: int,
    batchsize: int,
    num_attention_heads: int,
    num_key_value_heads: float,
    head_size: int,
    a_byte: float,
    kv_byte: float,
    qk_head_dim: int | None = None,
    v_head_dim: int | None = None,
) -> list[OperatorProfile]:
    """Standard (non-flash) prefill attention: returns [qk_matmul, sv_matmul, softmax]."""
    _qk = qk_head_dim if qk_head_dim is not None else head_size
    _v = v_head_dim if v_head_dim is not None else head_size
    is_mla = qk_head_dim is not None
    # MLA prefill: after kv_b_proj decompression, K/V are MHA (kv_heads = num_heads)
    kv_heads = num_attention_heads if is_mla else num_key_value_heads

    qk_ops = seqlen * seqlen * _qk * num_attention_heads * batchsize * 2
    sv_ops = seqlen * _v * seqlen * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * seqlen * seqlen * 5

    qk = OperatorProfile(
        name="qk_matmul",
        op_category="attention",
        flops=qk_ops,
        load_act=int(seqlen * _qk * batchsize * num_attention_heads * a_byte),
        store_act=int(seqlen * _qk * batchsize * num_attention_heads * a_byte),
        load_kv_cache=int(seqlen * _qk * batchsize * kv_heads * kv_byte),
    )
    sv = OperatorProfile(
        name="sv_matmul",
        op_category="attention",
        flops=sv_ops,
        load_act=int(seqlen * seqlen * batchsize * num_attention_heads * a_byte),
        store_act=int(seqlen * _v * batchsize * num_attention_heads * a_byte),
        load_kv_cache=int(seqlen * _v * batchsize * kv_heads * kv_byte),
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
    qk_head_dim: int | None = None,
    v_head_dim: int | None = None,
) -> list[OperatorProfile]:
    """FlashAttention-2 fused prefill attention."""
    _qk = qk_head_dim if qk_head_dim is not None else head_size
    _v = v_head_dim if v_head_dim is not None else head_size
    is_mla = qk_head_dim is not None

    qk_ops = seqlen * seqlen * _qk * num_attention_heads * batchsize * 2
    sv_ops = seqlen * _v * seqlen * num_attention_heads * batchsize * 2
    softmax_ops = batchsize * num_attention_heads * seqlen * seqlen * 5

    block_size_r = min(math.ceil(onchip_buffer / (kv_byte * _qk)), _qk)
    n_blocks_r = math.ceil(seqlen / block_size_r)
    q_numel = seqlen * _qk * batchsize * num_attention_heads * a_byte
    o_numel = seqlen * _v * batchsize * num_attention_heads * a_byte

    if is_mla:
        # MLA prefill: after kv_b_proj, K[qk_dim] and V[v_dim] are MHA (num_heads per TP)
        kv_io = int(n_blocks_r * seqlen * (_qk + _v) * batchsize * num_attention_heads * kv_byte)
    else:
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
