"""Phase 1 runtime contract types (op_plan §2).

Two-layer strongly-typed dynamic input, replacing the per-step reconstruction of
operators and the implicit ``**kwargs`` dict-passing between operator / backend /
DB schema:

- ``StepRuntime``: step-level dynamic input (phase, token counts, ...). Built once
  per step from ``StepShape``; passed to ``op.forward(step)``.
- ``OpRuntime``: a single operator's resolved runtime, partitioned into
  shape / parallel / runtime (mirrors ``OperatorSignature`` partitions so the DB
  signature and the roofline cost read the *same* field source).

模型图 ``forward(step)`` 构造 ``StepRuntime``, ``op.forward(runtime)`` 产 ``OpRuntime``,
CostRouter / backends 消费它 (统一 forward/StepOpPlan 路径)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from llm_infer_sim.core.graph.step_shape import StepShape


@dataclass(frozen=True)
class StepRuntime:
    """Step-level dynamic input. One per step, fed to ``op.forward(step)``."""

    phase: str
    total_tokens: int
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    max_prefill_seqlen: int = 0
    avg_decode_context_len: int = 0
    max_context_len: int = 0
    execution_mode: str = ""

    @classmethod
    def from_step(cls, step: "StepShape") -> "StepRuntime":
        return cls(
            phase=step.phase,
            total_tokens=step.total_tokens,
            num_prefill_tokens=step.num_prefill_tokens,
            num_decode_tokens=step.num_decode_tokens,
            num_prefill_requests=step.num_prefill_requests,
            num_decode_requests=step.num_decode_requests,
            max_prefill_seqlen=step.max_prefill_seqlen,
            avg_decode_context_len=step.avg_decode_context_len,
            max_context_len=step.max_context_len,
            execution_mode=step.execution_mode,
        )


@dataclass(frozen=True)
class OpRuntime:
    """A single operator's resolved runtime for this step.

    ``op_subtype`` is the *runtime* subtype that enters the signature
    (``OperatorSignature.op_subtype``). ``None`` means "use the operator's static
    default subtype"; operators whose subtype varies by step regime (e.g.
    attention prefill vs mixed_prefill) fill it explicitly.

    ``shape`` / ``parallel`` / ``runtime`` carry only the fields registered for
    this op_kind (see ``operator_schema.fields.SIGNATURE_FIELDS``); unknown keys
    are rejected when the signature is built, so a typo fails loudly instead of
    silently missing the DB.
    """

    phase: str
    op_subtype: str | None
    shape: Mapping[str, Any]
    parallel: Mapping[str, Any]
    runtime: Mapping[str, Any]
