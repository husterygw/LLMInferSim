"""真 vLLM attention backend 测时辅助 (FLASH_ATTN / FLASHINFER / etc).

调通后 attention runner 走 vLLM impl.forward(...) 真 kernel, 而不是 SDPA.
设计跟 AIC `collector/vllm/utils.py` + `collect_attn.py` 一致, 但只摘 BF16 +
non-MLA + non-FP8KV + no sliding window 这条最常用 path. AIC 是 Apache-2.0
modified from vLLM upstream `tests/v1/attention/utils.py`.

参数 contract:
    BatchSpec(seq_lens, query_lens)   ← 单 prefill: q_len == s_len 整段;
                                         decode:   q_len = 1, s_len = ctx + 1
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec

try:
    from vllm.utils import resolve_obj_by_qualname
except ImportError:
    from vllm.utils.import_utils import resolve_obj_by_qualname

try:
    from vllm.utils import cdiv
except ImportError:
    from vllm.utils.math_utils import cdiv


@dataclass
class BatchSpec:
    """Workload shape (per-seq) — driving CommonAttentionMetadata + kv cache."""
    seq_lens: list[int]     # total token count per seq (context + new)
    query_lens: list[int]   # new tokens per seq (1 for decode, full for prefill)

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)

    def total_query_tokens(self) -> int:
        return sum(self.query_lens)


class MockAttentionLayer:
    """impl.forward 拿 layer 取 q/k/v scale, 不实际算; 这里给 1.0 通过."""

    def __init__(self, device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


def create_vllm_config(
    *,
    max_model_len: int,
    block_size: int,
    num_gpu_blocks: int,
    max_num_seqs: int,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> VllmConfig:
    """构造测试用 VllmConfig. 不真正加载模型权重."""
    # opt-125m HF config 默认 max_position_embeddings=2048; 我们要测到 8192+,
    # 必须先 raise hf_config 上限, 再造 ModelConfig (后者会跟 hf_config 校验).
    # vLLM 提供 VLLM_ALLOW_LONG_MAX_MODEL_LEN 旁路, 但走 hf_config_override 更干净.
    import os as _os
    _os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

    model_config = ModelConfig(
        # facebook/opt-125m: 本地 HF cache, ~250MB; only meta info matters (不加载 weights).
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        max_model_len=max_model_len,
    )
    try:
        cache_config = CacheConfig(block_size=block_size, cache_dtype="auto", swap_space=0)
    except (TypeError, Exception):
        cache_config = CacheConfig(block_size=block_size, cache_dtype="auto")
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = 0

    parallel_config = ParallelConfig(tensor_parallel_size=1)
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max(8192, max_model_len),
        enable_chunked_prefill=True,
        max_model_len=max_model_len,
        is_encoder_decoder=False,
    )

    # mock methods backends may probe
    import types
    model_config.get_num_layers = types.MethodType(lambda self: 1, model_config)
    model_config.get_sliding_window_for_layer = types.MethodType(lambda self, i: None, model_config)
    model_config.get_logits_soft_cap_for_layer = types.MethodType(lambda self, i: 0.0, model_config)
    model_config.get_sm_scale_for_layer = types.MethodType(
        lambda self, i: 1.0 / model_config.get_head_size() ** 0.5, model_config,
    )
    # override head shape on hf_config + model_arch_config
    model_config.hf_config.head_dim = head_dim
    arch_cfg = getattr(model_config, "model_arch_config", None)
    if arch_cfg is not None and hasattr(arch_cfg, "head_size"):
        arch_cfg.head_size = head_dim
    model_config.hf_config.num_attention_heads = num_heads
    if arch_cfg is not None and hasattr(arch_cfg, "total_num_attention_heads"):
        arch_cfg.total_num_attention_heads = num_heads
    model_config.hf_config.num_key_value_heads = num_kv_heads
    if arch_cfg is not None and hasattr(arch_cfg, "total_num_kv_heads"):
        arch_cfg.total_num_kv_heads = num_kv_heads

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )


def create_common_attn_metadata(
    batch_spec: BatchSpec, block_size: int, device,
) -> CommonAttentionMetadata:
    query_start_loc = torch.zeros(batch_spec.batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens, dtype=torch.int32, device=device,
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = batch_spec.total_query_tokens()
    seq_lens_t = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens_t.cpu()
    max_seq_len = int(seq_lens_cpu.max())
    context_lens = [batch_spec.seq_lens[i] - batch_spec.query_lens[i]
                    for i in range(batch_spec.batch_size)]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.randint(
        0, 1000, (batch_spec.batch_size, max_blocks), dtype=torch.int32, device=device,
    )
    slot_mapping = torch.randint(0, 1000, (num_tokens,), dtype=torch.int64, device=device)
    max_query_len = max(batch_spec.query_lens)

    for kwargs in (
        dict(_seq_lens_cpu=seq_lens_cpu, _num_computed_tokens_cpu=num_computed_tokens_cpu),
        dict(seq_lens_cpu=seq_lens_cpu, num_computed_tokens_cpu=num_computed_tokens_cpu),
    ):
        try:
            return CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens_t,
                num_reqs=batch_spec.batch_size,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                block_table_tensor=block_table_tensor,
                slot_mapping=slot_mapping,
                causal=True,
                **kwargs,
            )
        except TypeError:
            continue
    raise RuntimeError("CommonAttentionMetadata signature not recognized")


def create_standard_kv_cache_spec(
    vllm_config: VllmConfig,
) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
        sliding_window=vllm_config.model_config.get_sliding_window(),
    )


def create_and_prepopulate_kv_cache(
    *,
    k_contexts: list[torch.Tensor],
    v_contexts: list[torch.Tensor],
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype,
    device,
    num_blocks: int,
    common_attn_metadata: CommonAttentionMetadata,
) -> torch.Tensor:
    batch_size = len(k_contexts)
    seq_lens = common_attn_metadata.seq_lens_cpu
    query_lens = (common_attn_metadata.query_start_loc_cpu[1:]
                  - common_attn_metadata.query_start_loc_cpu[:-1])
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    kv_cache = torch.empty(2, num_blocks, block_size, num_kv_heads, head_size,
                           dtype=dtype, device=device)
    kv_cache_flat = kv_cache.view(2, -1, num_kv_heads, head_size)

    start_block_idx = 1
    for i in range(batch_size):
        k_context, v_context = k_contexts[i], v_contexts[i]
        start = start_block_idx * block_size
        end = start + k_context.shape[0]
        kv_cache_flat[0, start:end, ...] = k_context
        kv_cache_flat[1, start:end, ...] = v_context
        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx
    perm = torch.randperm(blocks_end - 1) + 1
    inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
    inv_perm[1:] = torch.argsort(perm) + 1
    kv_cache[:, 1:blocks_end, ...] = kv_cache[:, perm, ...]

    start_block_idx = 1
    for i in range(batch_size):
        n = cdiv(int(seq_lens[i]), block_size)
        block_table[i, :n] = inv_perm[start_block_idx:start_block_idx + n]
        start_block_idx += n

    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = (
            block_table[i, block_indices] * block_size
            + token_inter_block_offsets.to(device)
        )

    return kv_cache


def resolve_backend(
    head_dim: int, dtype, block_size: int,
):
    """Use vLLM's platform selector; return (backend_name_str, builder_cls, impl_cls)."""
    cfg = AttentionSelectorConfig(
        head_size=head_dim, dtype=dtype,
        kv_cache_dtype=None, block_size=block_size,
    )
    qualname = current_platform.get_attn_backend_cls(None, cfg)
    backend_class = resolve_obj_by_qualname(qualname)
    return (
        backend_class.get_name(),
        backend_class.get_builder_cls(),
        backend_class.get_impl_cls(),
    )
