"""GlobalStepWorkload — 框架无关的 step workload 描述。

阶段一: 字段够用即可, 不引入 KV cache 状态 / preemption / spec decode 等。
后续阶段会按需补字段。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StepPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"
    CHUNKED_PREFILL = "chunked_prefill"


@dataclass
class RequestWorkload:
    """单个请求在本 step 中的工作负载。"""
    request_id: str
    phase: StepPhase
    num_tokens: int                # 本 step 调度的 token 数
    context_len: int = 0           # 当前上下文长度 (含历史)
    target_output_len: int = 0     # 目标输出长度 (sampling_params.max_tokens)
    generated_tokens: int = 0      # 已生成的 output token 数
    is_chunked: bool = False
    chunk_size: int = 0


@dataclass
class GlobalStepWorkload:
    """一个 scheduler step 的全局工作负载描述。"""
    step_id: int
    phase: StepPhase
    requests: list[RequestWorkload] = field(default_factory=list)

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    total_scheduled_tokens: int = 0
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    # 本 step 中新到 prefill 请求已被 prefix-cache 命中, 因此本不需 forward 的 token 数。
    # 仅统计 new_req.num_computed_tokens > 0 的情形 (cache 命中); 已存在的 cached_req
    # 的 num_computed_tokens 来自先前 step 自然累加, 不算 prefix caching 节省。
    num_prefix_cached_tokens: int = 0

    @property
    def max_context_len(self) -> int:
        return max((r.context_len for r in self.requests), default=0)

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def avg_decode_context_len(self) -> int:
        """阶段 3: MixedAttentionEstimator 需要 decode 段的平均 ctx_len。"""
        decode_ctxs = [
            r.context_len for r in self.requests if r.phase == StepPhase.DECODE
        ]
        if not decode_ctxs:
            return 0
        return sum(decode_ctxs) // len(decode_ctxs)

    @property
    def max_prefill_seqlen(self) -> int:
        """Mixed step 中 prefill 段的最长 seqlen (本 step 调度的 prefill token 数)。"""
        prefill_lens = [
            r.num_tokens for r in self.requests
            if r.phase in (StepPhase.PREFILL, StepPhase.CHUNKED_PREFILL)
        ]
        return max(prefill_lens, default=0)
