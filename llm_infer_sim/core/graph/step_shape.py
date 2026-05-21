"""V3 §4.2 StepShape — graph/cost 层消费的 step shape.

从 GlobalStepWorkload 派生, 加入 graph/cudagraph 和 cost 相关字段.

阶段 1:    PREFILL / DECODE.
阶段 3d:   MIXED — prefill seqs + decode seqs 同 step (chunked prefill).
            num_prefill_tokens > 0 AND num_decode_requests > 0 → phase="mixed".
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.workload.workload import GlobalStepWorkload, StepPhase


@dataclass(frozen=True)
class StepShape:
    step_id: int
    phase: str
    total_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    num_prefill_requests: int
    num_decode_requests: int
    max_context_len: int
    max_prefill_seqlen: int
    avg_decode_context_len: int
    execution_mode: str
    graph_capture_size: int | None = None
    padded_tokens: int | None = None

    @classmethod
    def from_workload(
        cls,
        workload: GlobalStepWorkload,
        deploy: DeployConfig,
    ) -> StepShape:
        phase_str = (
            workload.phase.value
            if isinstance(workload.phase, StepPhase)
            else str(workload.phase)
        )
        if phase_str not in ("prefill", "decode", "mixed", "chunked_prefill"):
            raise NotImplementedError(
                f"StepShape only supports prefill / decode / mixed / chunked_prefill, "
                f"got {phase_str!r}."
            )
        return cls(
            step_id=workload.step_id,
            phase=phase_str,
            total_tokens=workload.total_scheduled_tokens,
            num_prefill_tokens=workload.num_prefill_tokens,
            num_decode_tokens=workload.num_decode_tokens,
            num_prefill_requests=workload.num_prefill_requests,
            num_decode_requests=workload.num_decode_requests,
            max_context_len=workload.max_context_len,
            max_prefill_seqlen=workload.max_prefill_seqlen,
            avg_decode_context_len=workload.avg_decode_context_len,
            execution_mode=deploy.execution_mode,
        )
