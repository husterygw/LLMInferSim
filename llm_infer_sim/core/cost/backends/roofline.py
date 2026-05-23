"""RooflineBackend — Operator -> roofline latency."""
from __future__ import annotations

from typing import Any

from llm_infer_sim.core.cost.roofline_analyzer import RooflineAnalyzer
from llm_infer_sim.core.cost.trace import CostTraceEntry, format_display_name
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig


class RooflineBackend:
    """阶段 1: 唯一 cost backend."""

    def __init__(
        self,
        hw: HardwareConfig,
        deploy: DeployConfig,
        *,
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: int = 16,
        efficiency_profile=None,
    ):
        self.hw = hw
        self.deploy = deploy
        self.analyzer = RooflineAnalyzer(
            hw,
            w_bit=w_bit,
            a_bit=a_bit,
            kv_bit=kv_bit,
            efficiency_profile=efficiency_profile,
            execution_mode=deploy.execution_mode,
        )

    def estimate(self, op: Any) -> CostTraceEntry:
        spec = self._roofline_spec(op)
        result = self.analyzer.analyze(op.name, spec)
        return CostTraceEntry(
            op_name=op.name,
            op_kind=op.op_kind,
            op_subtype=op.op_subtype,
            latency_s=result.total_time,
            source="roofline",
            match_type="fallback",
            layer_idx=getattr(op, "layer_idx", None),
            display_name=format_display_name(op.name, getattr(op, "layer_idx", None)),
            roofline_s=result.total_time,
            roofline_gap=None,
            metadata={
                "bottleneck": result.bottleneck,
                "t_compute": result.t_compute,
                "t_memory": result.t_memory,
                "t_comm": result.t_comm,
                "kernel_overhead": result.kernel_overhead,
                "arithmetic_intensity": result.arithmetic_intensity,
                "achievable_performance": result.achievable_performance,
                "mem_bytes": result.mem_bytes,
                "flops": result.flops,
            },
        )

    @staticmethod
    def _roofline_spec(op: Any) -> RooflineSpec:
        spec_attr = getattr(op, "roofline_spec")
        spec = spec_attr() if callable(spec_attr) else spec_attr
        if isinstance(spec, RooflineSpec):
            return spec

        f = spec
        return RooflineSpec(
            op_category=f.get("op_category", "matmul"),
            flops=int(f.get("flops", 0)),
            load_weight=int(f.get("load_weight", 0)),
            load_act=int(f.get("load_act", 0)),
            store_act=int(f.get("store_act", 0)),
            load_kv_cache=int(f.get("load_kv_cache", 0)),
            store_kv_cache=int(f.get("store_kv_cache", 0)),
            op_precision=str(f.get("op_precision", "")),
            comm_bytes=float(f.get("comm_bytes", 0.0)),
            comm_type=str(f.get("comm_type", "")),
        )
