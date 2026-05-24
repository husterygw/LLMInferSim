"""CommunicationRooflineBackend.

Collective ops are modeled with the low-level NCCL-style roofline primitives
under ``core.cost.roofline.communication``.  OperatorDB exact hit for
collectives is intentionally left for a later calibration phase.
"""
from __future__ import annotations

import os
from typing import Any

from llm_infer_sim.core.cost.roofline import communication as comm
from llm_infer_sim.core.cost.trace import CostTraceEntry, format_display_name
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig


class CommunicationRooflineBackend:
    """Estimate Collective operators with communication roofline primitives."""

    def __init__(self, hw: HardwareConfig, deploy: DeployConfig):
        self.hw = hw
        self.deploy = deploy

    def estimate(self, op: Any) -> CostTraceEntry:
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
        latency_s = self._estimate_latency(
            comm_type=comm_type,
            message_bytes=message_bytes,
            world_size=world_size,
            execution_mode=execution_mode,
            topology_hint=topology_hint,
            algo=str(runtime.get("algo", "auto") or "auto"),
        )

        metadata = {
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

    def _estimate_latency(
        self,
        *,
        comm_type: str,
        message_bytes: int,
        world_size: int,
        execution_mode: str,
        topology_hint: str,
        algo: str,
    ) -> float:
        kwargs = {
            "mode": execution_mode,
            "topology_hint": topology_hint,
        }
        if comm_type == "allreduce":
            return comm.allreduce_time(
                message_bytes, world_size, self.hw, algo=algo, **kwargs,
            )
        if comm_type == "allgather":
            return comm.allgather_time(
                message_bytes, world_size, self.hw, algo=algo, **kwargs,
            )
        if comm_type in ("reducescatter", "reduce_scatter"):
            return comm.reducescatter_time(
                message_bytes, world_size, self.hw, **kwargs,
            )
        if comm_type == "alltoall":
            return comm.alltoall_time(
                message_bytes, world_size, self.hw, algo=algo, **kwargs,
            )
        if comm_type == "broadcast":
            return comm.broadcast_time(message_bytes, world_size, self.hw, **kwargs)
        if comm_type == "p2p":
            return comm.p2p_time(message_bytes, self.hw, **kwargs)
        raise NotImplementedError(f"unsupported collective type: {comm_type!r}")

    @staticmethod
    def _topology_hint(runtime: dict[str, Any]) -> str:
        topology = str(runtime.get("topology", "") or "")
        if topology in ("concentrated", "balanced"):
            return topology
        return os.environ.get("LLM_INFER_SIM_NUMA_HINT", "concentrated")
