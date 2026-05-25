"""RooflineBackend — Operator -> roofline latency.

唯一 roofline backend. 内部按 op_kind 分发到 compute (GEMM/Attention/MoE/etc.)
或 collective (AllReduce/AllToAll/...) roofline model.

comm_plan Step 3: collective dispatch 从外部 CommunicationRooflineBackend 收回内部,
CostRouter 不再特判 op_kind == "collective".
"""
from __future__ import annotations

import os
from typing import Any

from llm_infer_sim.core.cost.roofline import communication as comm
from llm_infer_sim.core.cost.roofline_analyzer import RooflineAnalyzer
from llm_infer_sim.core.cost.trace import CostTraceEntry, format_display_name
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig


class RooflineBackend:
    """唯一 roofline backend (compute + collective)."""

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
        if op.op_kind == "collective":
            return self._estimate_collective(op)
        return self._estimate_compute(op)

    def _estimate_compute(self, op: Any) -> CostTraceEntry:
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

    def _estimate_collective(self, op: Any) -> CostTraceEntry:
        """Collective op latency: 调 core/cost/roofline/communication.py 公式.

        Step 3 接管自旧 CommunicationRooflineBackend.
        Step 4: AllReduce 优先走 hw.communication.backends["nccl"].allreduce.* 新参数 +
            ll_tree / ll128_tree / simple_ring / simple_tree 候选, 候选 breakdown
            写入 metadata.allreduce_candidates / selected_algorithm.
        Step 6-A: AllToAll 同样走 hw.communication.backends["nccl"].alltoall.* +
            pairwise 候选 (with contention_factor), breakdown 写
            metadata.alltoall_candidates / selected_algorithm.
        """
        comm_type = str(op.op_subtype)
        message_bytes = int(getattr(op, "message_bytes", 0))
        if message_bytes <= 0:
            message_bytes = int(op.shape.get("message_bytes", 0))
        world_size = int(getattr(op, "world_size", 1))
        if world_size <= 1:
            world_size = int(op.parallel.get("world_size", 1))

        runtime = op.runtime
        execution_mode = str(runtime.get("execution_mode", self.deploy.execution_mode))
        topology_hint = self._topology_hint(runtime)
        algo_hint = str(runtime.get("algo", "auto") or "auto")
        protocol_hint = runtime.get("protocol_hint")
        latency_s, breakdown = self._dispatch_collective(
            comm_type=comm_type,
            message_bytes=message_bytes,
            world_size=world_size,
            execution_mode=execution_mode,
            topology_hint=topology_hint,
            algo=algo_hint,
            protocol_hint=protocol_hint,
        )

        metadata: dict[str, Any] = {
            "bottleneck": "communication",
            "t_compute": 0.0,
            "t_memory": 0.0,
            "t_comm": latency_s,
            "comm_type": comm_type,
            "message_bytes": message_bytes,
            "world_size": world_size,
            "topology_hint": topology_hint,
            "execution_mode": execution_mode,
            "backend": runtime.get("backend", "nccl"),
        }
        if breakdown:
            metadata["selected_algorithm"] = breakdown.get("selected")
            metadata["algorithm_term_s"] = breakdown.get("algorithm_term")
            metadata["framework_overhead_s"] = breakdown.get(
                "framework_overhead_s", 0.0
            )
            metadata["communication_path"] = breakdown.get("path")
            cands = breakdown.get("candidates") or {}
            if cands:
                metadata[f"{comm_type}_candidates"] = dict(cands)

        return CostTraceEntry(
            op_name=op.name,
            op_kind=op.op_kind,
            op_subtype=op.op_subtype,
            latency_s=latency_s,
            source="comm_roofline",
            match_type="fallback",
            layer_idx=getattr(op, "layer_idx", None),
            display_name=format_display_name(op.name, getattr(op, "layer_idx", None)),
            roofline_s=latency_s,
            roofline_gap=None,
            metadata=metadata,
        )

    def _dispatch_collective(
        self,
        *,
        comm_type: str,
        message_bytes: int,
        world_size: int,
        execution_mode: str,
        topology_hint: str,
        algo: str,
        protocol_hint: Any = None,
    ) -> tuple[float, dict[str, Any]]:
        """返回 (latency_s, breakdown). breakdown 仅 allreduce 非空."""
        kwargs = {"mode": execution_mode, "topology_hint": topology_hint}
        if comm_type == "allreduce":
            return comm.allreduce_time_with_breakdown(
                message_bytes, world_size, self.hw,
                algo=algo, protocol_hint=protocol_hint, **kwargs,
            )
        if comm_type == "allgather":
            return comm.allgather_time(
                message_bytes, world_size, self.hw, algo=algo, **kwargs,
            ), {}
        if comm_type in ("reducescatter", "reduce_scatter"):
            return comm.reducescatter_time(
                message_bytes, world_size, self.hw, **kwargs,
            ), {}
        if comm_type == "alltoall":
            return comm.alltoall_time_with_breakdown(
                message_bytes, world_size, self.hw, algo=algo, **kwargs,
            )
        if comm_type == "broadcast":
            return comm.broadcast_time(
                message_bytes, world_size, self.hw, **kwargs,
            ), {}
        if comm_type == "p2p":
            return comm.p2p_time(message_bytes, self.hw, **kwargs), {}
        raise NotImplementedError(f"unsupported collective type: {comm_type!r}")

    @staticmethod
    def _topology_hint(runtime: dict[str, Any]) -> str:
        topology = str(runtime.get("topology", "") or "")
        if topology in ("concentrated", "balanced"):
            return topology
        return os.environ.get("LLM_INFER_SIM_NUMA_HINT", "concentrated")

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
