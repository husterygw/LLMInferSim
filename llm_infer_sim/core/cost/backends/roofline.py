"""RooflineBackend — Operator -> roofline latency.

唯一 roofline backend. 内部按 op_kind 分发到 compute (GEMM/Attention/MoE/etc.)
或 collective (AllReduce/AllToAll/...) roofline model.

comm_plan Step 3: collective dispatch 从外部 CommunicationRooflineBackend 收回内部,
CostRouter 不再特判 op_kind == "collective".
"""
from __future__ import annotations

import os
from typing import Any

from llm_infer_sim.core.calibration.profile import CalibrationProfile
from llm_infer_sim.core.cost.roofline import communication as comm
from llm_infer_sim.core.cost.roofline_analyzer import RooflineAnalyzer
from llm_infer_sim.core.cost.trace import CostTraceEntry, format_display_name
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.hardware.device import HardwareProfile

# collective comm_type 集合 (_dispatch_collective 支持的). moe_dispatch 解析为本地时
# comm_type 不在此集合 (None/pre_dispatch) → 通信成本 0.
_COLLECTIVE_TYPES = frozenset({
    "allreduce", "allgather", "reducescatter", "reduce_scatter",
    "alltoall", "broadcast", "p2p",
})


class RooflineBackend:
    """唯一 roofline backend (compute + collective)."""

    def __init__(
        self,
        hw: HardwareProfile,
        execution_mode: str = "eager",
        *,
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: int = 16,
        calibration: CalibrationProfile | None = None,
    ):
        self.hw = hw
        self.execution_mode = execution_mode
        self.calibration = calibration or CalibrationProfile()
        self.analyzer = RooflineAnalyzer(
            hw,
            w_bit=w_bit,
            a_bit=a_bit,
            kv_bit=kv_bit,
            execution_mode=execution_mode,
            calibration=self.calibration,
        )

    def estimate(self, op: Any, op_runtime: Any = None) -> CostTraceEntry:
        # op_runtime (OpRuntime | None): None = legacy (op carries dynamic params
        # from construction). A non-None op_runtime (from op.forward(step)) is
        # consumed by ops migrated to the static contract (Phase 2: GEMM).
        # collective still reads op properties (migrates in a later phase).
        # moe_dispatch 现承载 dispatch 通信 (AIC 对齐: MoEDispatch=通信), 走 collective 路径.
        if op.op_kind in ("collective", "moe_dispatch"):
            return self._estimate_collective(op, op_runtime)
        return self._estimate_compute(op, op_runtime)

    def _estimate_compute(self, op: Any, op_runtime: Any = None) -> CostTraceEntry:
        spec = self._roofline_spec(op, op_runtime)
        result = self.analyzer.analyze(op.name, spec)
        metadata: dict[str, Any] = {
            "bottleneck": result.bottleneck,
            "t_compute": result.t_compute,
            "t_memory": result.t_memory,
            "t_comm": result.t_comm,
            "kernel_overhead": result.kernel_overhead,
            "arithmetic_intensity": result.arithmetic_intensity,
            "achievable_performance": result.achievable_performance,
            "mem_bytes": result.mem_bytes,
            "flops": result.flops,
        }
        # moe_plan §5.A: MoE calibration 后处理. 3 类 op 各按 knob 修 latency:
        #   moe_topk           latency += topk_overhead_us
        #   moe_dispatch       latency += local_dispatch_overhead_us
        #   moe (routed_*)     latency /= grouped_gemm_efficiency (<1 慢, >1 快)
        latency_s = self._apply_moe_calibration(op, result.total_time, metadata)

        # build-once: the resolved subtype (e.g. mixed_prefill/mixed_decode) lives
        # on op_runtime; the static op carries only the placeholder (prefill/decode).
        op_subtype = (op_runtime.op_subtype if op_runtime is not None and op_runtime.op_subtype
                      else op.op_subtype)
        return CostTraceEntry(
            op_name=op.name,
            op_kind=op.op_kind,
            op_subtype=op_subtype,
            latency_s=latency_s,
            source="roofline",
            match_type="fallback",
            layer_idx=getattr(op, "layer_idx", None),
            display_name=format_display_name(op.name, getattr(op, "layer_idx", None)),
            roofline_s=result.total_time,
            roofline_gap=None,
            metadata=metadata,
        )

    def _apply_moe_calibration(
        self, op: Any, roofline_latency_s: float, metadata: dict[str, Any],
    ) -> float:
        """moe_plan §5.A: apply MoE calibration knobs + write metadata.

        每个 knob 必须只影响对应 latency term (plan §5.A 验收 step 4).
        无 calibration (calibration.moe_efficiency is None) 时 latency_s = roofline_latency_s
        且不写 t_topk / t_dispatch_local / t_expert_compute / grouped_gemm_efficiency
        / moe_profile_id 等字段.
        """
        prof = self.calibration.moe_efficiency
        if prof is None:
            return roofline_latency_s

        op_kind = op.op_kind
        op_subtype = getattr(op, "op_subtype", "")
        latency = roofline_latency_s
        applied = False

        if op_kind == "elementwise" and op_subtype == "topk":
            overhead_s = float(prof.topk_overhead_us) * 1e-6
            latency = roofline_latency_s + overhead_s
            metadata["t_topk"] = latency
            metadata["topk_overhead_us"] = prof.topk_overhead_us
            applied = True
        elif op_kind == "moe_dispatch":
            overhead_s = float(prof.local_dispatch_overhead_us) * 1e-6
            latency = roofline_latency_s + overhead_s
            metadata["t_dispatch_local"] = latency
            metadata["local_dispatch_overhead_us"] = prof.local_dispatch_overhead_us
            applied = True
        elif op_kind == "moe":
            eff = float(prof.grouped_gemm_efficiency)
            if eff <= 0:
                eff = 1.0
            latency = roofline_latency_s / eff
            metadata["t_expert_compute"] = latency
            metadata["grouped_gemm_efficiency"] = eff
            applied = True

        if applied:
            metadata["moe_profile_id"] = prof.profile_id
        return latency

    def _estimate_collective(self, op: Any, op_runtime: Any = None) -> CostTraceEntry:
        """Collective op latency: 调 core/cost/roofline/communication.py 公式.

        Step 3 接管自旧 CommunicationRooflineBackend.
        Step 4: AllReduce 优先走 hw.communication.backends["nccl"].allreduce.* 新参数 +
            ll_tree / ll128_tree / simple_ring / simple_tree 候选, 候选 breakdown
            写入 metadata.allreduce_candidates / selected_algorithm.
        Step 6-A: AllToAll 同样走 hw.communication.backends["nccl"].alltoall.* +
            pairwise 候选 (with contention_factor), breakdown 写
            metadata.alltoall_candidates / selected_algorithm.
        """
        if op_runtime is not None:
            # migrated static collective: read shape/parallel/runtime from the
            # step-resolved OpRuntime (legacy reads them off the op directly).
            comm_type = str(op_runtime.op_subtype or op.op_subtype)
            message_bytes = int(op_runtime.shape.get("message_bytes", 0))
            world_size = int(op_runtime.parallel.get("world_size", 1))
            runtime = op_runtime.runtime
        else:
            comm_type = str(op.op_subtype)
            message_bytes = int(getattr(op, "message_bytes", 0))
            if message_bytes <= 0:
                message_bytes = int(op.shape.get("message_bytes", 0))
            world_size = int(getattr(op, "world_size", 1))
            if world_size <= 1:
                world_size = int(op.parallel.get("world_size", 1))
            runtime = op.runtime
        execution_mode = str(runtime.get("execution_mode", self.execution_mode))
        topology_hint = self._topology_hint(runtime)
        algo_hint = str(runtime.get("algo", "auto") or "auto")
        protocol_hint = runtime.get("protocol_hint")
        # MoEDispatch 解析为本地 (无跨卡通信) 时 comm_type=None/pre_dispatch、bytes=0
        # 或 world<=1 → 通信成本 0 (跳过 _dispatch_collective, 避免对非 collective
        # comm_type 报 NotImplementedError); local_dispatch_overhead 由下面 calibration 加.
        if comm_type in _COLLECTIVE_TYPES and message_bytes > 0 and world_size > 1:
            latency_s, breakdown = self._dispatch_collective(
                comm_type=comm_type,
                message_bytes=message_bytes,
                world_size=world_size,
                execution_mode=execution_mode,
                topology_hint=topology_hint,
                algo=algo_hint,
                protocol_hint=protocol_hint,
            )
        else:
            latency_s, breakdown = 0.0, {}

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

        # MoE calibration: moe_dispatch op 在 comm 之上叠加 local_dispatch_overhead_us
        # (本地 permute/align). 真 collective (op_kind=collective) 无匹配分支 → 不变.
        comm_latency_s = latency_s
        latency_s = self._apply_moe_calibration(op, latency_s, metadata)

        return CostTraceEntry(
            op_name=op.name,
            op_kind=op.op_kind,
            op_subtype=op.op_subtype,
            latency_s=latency_s,
            source="comm_roofline",
            match_type="fallback",
            layer_idx=getattr(op, "layer_idx", None),
            display_name=format_display_name(op.name, getattr(op, "layer_idx", None)),
            roofline_s=comm_latency_s,
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
    def _roofline_spec(op: Any, op_runtime: Any = None) -> RooflineSpec:
        spec_attr = getattr(op, "roofline_spec")
        if callable(spec_attr):
            # migrated ops accept op_runtime; pass it only when present so legacy
            # no-arg roofline_spec() signatures are unaffected.
            spec = spec_attr(op_runtime) if op_runtime is not None else spec_attr()
        else:
            spec = spec_attr
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
