"""OperatorDBBackend — V3 §7.1 / IMPL_PLAN §3.6.

Operator -> OperatorSignature -> store.lookup
    hit  -> CostTraceEntry(source=operator_db, match_type=exact, latency_s=real)
    miss -> None  (CostRouter 决定 fallback)

Stage 3 范围: GEMM exact hit; attention/moe/collective canonicalizer 已具备,
但 measured data 还没采全, 大部分会 miss (fallback to RooflineBackend).
"""
from __future__ import annotations

from typing import Any

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.trace import CostTraceEntry
from llm_infer_sim.core.operator_db.store import OperatorStore
from llm_infer_sim.core.operator_schema import operator_to_signature


class OperatorDBBackend:
    """Read-only DB lookup. 不做 fallback (那是 CostRouter 的事)."""

    SUPPORTED_KINDS = ("gemm", "attention", "moe", "collective")

    def __init__(
        self,
        store: OperatorStore,
        *,
        roofline: RooflineBackend | None = None,
    ):
        """roofline 可选: 若给, hit entry 携带 roofline_s + roofline_gap (V3 §4.5)."""
        self.store = store
        self.roofline = roofline

    def estimate(self, op: Any) -> CostTraceEntry | None:
        if op.op_kind not in self.SUPPORTED_KINDS:
            return None
        try:
            signature = operator_to_signature(op)
        except ValueError:
            return None
        record = self.store.lookup(signature)
        if record is None:
            return None

        latency_s = record.latency_s
        roofline_s, roofline_gap = self._roofline_compare(op, latency_s)
        return CostTraceEntry(
            op_name=op.name,
            op_kind=op.op_kind,
            op_subtype=op.op_subtype,
            latency_s=latency_s,
            source="operator_db",
            match_type="exact",
            roofline_s=roofline_s,
            roofline_gap=roofline_gap,
            metadata={
                "case_id": record.source.get("case_id"),
                "kernel_source": record.kernel_source,
                "hardware": record.hardware,
                "framework_version": record.framework_version,
                "latency_us_p50": record.latency_us_p50,
                "latency_us_p10": record.latency_us_p10,
                "latency_us_p90": record.latency_us_p90,
                "n_iters": record.n_iters,
                "confidence": record.confidence,
            },
        )

    def _roofline_compare(
        self, op: Any, real_latency_s: float,
    ) -> tuple[float | None, float | None]:
        """跑 roofline 拿 lower bound, 用 real / roofline 算 gap."""
        if self.roofline is None:
            return None, None
        rl_entry = self.roofline.estimate(op)
        if rl_entry.latency_s <= 0:
            return rl_entry.latency_s, None
        return rl_entry.latency_s, real_latency_s / rl_entry.latency_s
