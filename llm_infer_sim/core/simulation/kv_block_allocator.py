"""KVBlockAllocator — 详设 §10.5 4.5 + §7.6 PD 分离共享基建。

vLLM 内部用 PagedAttention BlockAllocator 管 KV cache:
  - 固定大小 block (默认 16 tok), 共 num_blocks 个
  - prefix cache 命中: 多请求共享 block 不占新内存
  - 满了 → eviction (LRU) / preempt seq

我们这层是**观察者**, 不真分配内存。每 step 看 scheduler_output 增量统计:
  - new_req 进来: ceil(prompt_len / block_size) 总需要量, 减去 cached blocks = 新分配
  - cached_req decode 跨 block 边界: +1 block
  - finished_req_ids: 释放该 req 全部 block
  - 跨 step 维护 per_req_blocks 字典

用途:
  1. **prefix cache 命中率聚合**: cached_tokens / total_prompt_tokens
  2. **block-level memory utilization**: in_use_blocks / total_blocks (诊断/未来 preempt)
  3. **PD 分离 KV 传输 cost**: producer prefill 完成时, 该 req block 数 × block_bytes
     = 跨节点 NCCL/RDMA 传输 bytes
  4. **MLA / MHA / DSA 不同 block_bytes 公式区分**

非目标 (当前 MVP 不做):
  - 真 LRU eviction (vLLM 自己做了, 我们 cost model 当作 fixed budget)
  - block 内 fragmentation (vLLM 内部)
  - cross-request block sharing graph (太细)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.models.config import ModelProfile


def compute_block_bytes(
    model: ModelProfile,
    block_size: int,
    kv_byte: float,
) -> int:
    """单 block 的字节数 (跨所有 layer 累加, 含 K+V 或 c_kv).

    MLA (kv_lora_rank > 0):
        每 token / 层 = (kv_lora_rank + rope_head_dim) × kv_byte
        无 K+V × 2: MLA 只存 c_kv 单张量, K/V 通过 absorbed 矩阵投影出来

    标准 MHA / GQA:
        每 token / 层 = num_kv_heads × head_dim × 2 (K, V) × kv_byte

    阶段 9 V4 sparse / V3.2 DSA: 主 KV cache 走 MLA 公式; lightning indexer
    cache 独立, 暂不计入 (规模 ~indexer_cache_bytes / 主 cache <5%, MVP 简化)。
    """
    L = model.num_layers
    if model.kv_lora_rank > 0:
        per_tok_per_layer = (model.kv_lora_rank + model.rope_head_dim) * kv_byte
    else:
        per_tok_per_layer = model.num_kv_heads * model.head_dim * 2 * kv_byte
    return int(block_size * L * per_tok_per_layer)


@dataclass
class StepBlockStats:
    """每 step allocator 的增量统计."""
    new_blocks_allocated: int = 0        # 本 step 真正占新 block (扣 dedup)
    blocks_dedup_hit: int = 0            # 本 step 因 prefix cache 命中省下的 block
    blocks_freed: int = 0                # 本 step finished_req 释放的 block
    blocks_in_use_after: int = 0         # step 结束后的总 in-use block 数
    blocks_available_after: int = 0      # = total - in_use_after (可正可负, 负=超额)
    memory_in_use_bytes: int = 0
    cached_tokens_this_step: int = 0
    new_prompt_tokens_this_step: int = 0


@dataclass
class CumulativeBlockStats:
    """跨 step 累积统计 (供 aggregate 报告)."""
    cumulative_cached_tokens: int = 0
    cumulative_total_prompt_tokens: int = 0
    cumulative_new_blocks_allocated: int = 0
    cumulative_blocks_dedup_hit: int = 0
    cumulative_blocks_freed: int = 0
    peak_blocks_in_use: int = 0
    num_steps_over_capacity: int = 0     # in_use > total 的 step 数 (超容量计数)

    @property
    def prefix_cache_hit_rate(self) -> float:
        """命中率 = cached / total_prompt; 无 prompt 时返 0."""
        if self.cumulative_total_prompt_tokens == 0:
            return 0.0
        return self.cumulative_cached_tokens / self.cumulative_total_prompt_tokens

    @property
    def block_dedup_hit_rate(self) -> float:
        """block 级 dedup 命中率 = dedup_hit / (dedup_hit + new_allocated)."""
        total = self.cumulative_blocks_dedup_hit + self.cumulative_new_blocks_allocated
        if total == 0:
            return 0.0
        return self.cumulative_blocks_dedup_hit / total


class KVBlockAllocator:
    """观察者: 跟 vLLM 内部 BlockAllocator 平行的轻量账本, 仅追踪 alloc/free 数量.

    Args:
        model: 模型配置 (决 block_bytes 公式 — MLA vs MHA)
        block_size: vLLM cache_config.block_size (默认 16)
        num_blocks_total: vLLM determine_available_memory 算出来的总 block 数
        kv_byte: KV cache dtype 字节 (1.0 fp8 / 2.0 fp16 / 0.5 fp4)
    """

    def __init__(
        self,
        model: ModelProfile,
        block_size: int,
        num_blocks_total: int,
        kv_byte: float = 2.0,
    ) -> None:
        self.model = model
        self.block_size = block_size
        self.num_blocks_total = num_blocks_total
        self.kv_byte = kv_byte
        self.block_bytes = compute_block_bytes(model, block_size, kv_byte)

        # state
        self._req_blocks: dict[str, int] = {}   # req_id → 该 req 当前持有的 block 数
        self.cumulative = CumulativeBlockStats()

    def step(self, scheduler_output: Any, num_prefix_cached_tokens: int) -> StepBlockStats:
        """处理一个 step, 返回增量统计 + 更新 cumulative.

        Args:
            scheduler_output: vLLM SchedulerOutput-shape (含 new_reqs / cached / finished)
            num_prefix_cached_tokens: step_extractor 已算好的本 step prefix cache 命中数

        Returns:
            StepBlockStats: 本 step 增量
        """
        new_blocks_allocated = 0
        blocks_dedup_hit = 0
        new_prompt_tokens = 0

        # ---- 1. new_req: 全 prompt 需求 - cached 命中 = 新分配 ----
        for new_req in scheduler_output.scheduled_new_reqs:
            prompt_len = len(new_req.prompt_token_ids or [])
            already_computed = new_req.num_computed_tokens
            blocks_needed = _ceil_div(prompt_len, self.block_size)
            # vLLM 实际命中以 block 对齐: 部分 block 不算命中
            cached_blocks = already_computed // self.block_size
            allocated = max(blocks_needed - cached_blocks, 0)

            self._req_blocks[new_req.req_id] = blocks_needed
            new_blocks_allocated += allocated
            blocks_dedup_hit += cached_blocks
            new_prompt_tokens += prompt_len

        # ---- 2. cached_req: decode 跨 block 边界时新增 ----
        # decode 每 step 多 1 tok, 当 (num_computed+1) % block_size == 0 时就跨边界。
        # 但 num_scheduled_tokens 可能 > 1 (chunked prefill 续段), 此时按 tok 数算。
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            num_computed_before = cached.num_computed_tokens[i]
            ntok = scheduler_output.num_scheduled_tokens.get(rid, 1)
            current_blocks = self._req_blocks.get(rid, 0)
            # 跨 step 后该 req 总占 token 数 = num_computed_before + ntok
            needed_blocks = _ceil_div(num_computed_before + ntok, self.block_size)
            delta = max(needed_blocks - current_blocks, 0)
            self._req_blocks[rid] = max(current_blocks, needed_blocks)
            new_blocks_allocated += delta

        # ---- 3. finished_req_ids: 释放该 req 全部 block ----
        blocks_freed = 0
        for fid in scheduler_output.finished_req_ids:
            if fid in self._req_blocks:
                blocks_freed += self._req_blocks.pop(fid)

        in_use = sum(self._req_blocks.values())
        available = self.num_blocks_total - in_use

        # ---- 4. 更新 cumulative ----
        c = self.cumulative
        c.cumulative_cached_tokens += num_prefix_cached_tokens
        c.cumulative_total_prompt_tokens += new_prompt_tokens
        c.cumulative_new_blocks_allocated += new_blocks_allocated
        c.cumulative_blocks_dedup_hit += blocks_dedup_hit
        c.cumulative_blocks_freed += blocks_freed
        if in_use > c.peak_blocks_in_use:
            c.peak_blocks_in_use = in_use
        if in_use > self.num_blocks_total:
            c.num_steps_over_capacity += 1

        return StepBlockStats(
            new_blocks_allocated=new_blocks_allocated,
            blocks_dedup_hit=blocks_dedup_hit,
            blocks_freed=blocks_freed,
            blocks_in_use_after=in_use,
            blocks_available_after=available,
            memory_in_use_bytes=in_use * self.block_bytes,
            cached_tokens_this_step=num_prefix_cached_tokens,
            new_prompt_tokens_this_step=new_prompt_tokens,
        )

    def get_req_blocks(self, req_id: str) -> int:
        """该 req 当前持有的 block 数 (PD 分离 producer 用: 序列化 send 张量)."""
        return self._req_blocks.get(req_id, 0)

    def req_kv_bytes(self, req_id: str) -> int:
        """该 req 当前持有的 KV 字节数 (PD transfer 用)."""
        return self.get_req_blocks(req_id) * self.block_bytes


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        return 0
    return -(-a // b)
