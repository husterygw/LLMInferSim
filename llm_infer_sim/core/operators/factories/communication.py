"""CollectiveOpFactory — V3 §6.5 / IMPL_PLAN §5.

阶段 3a-3e 范围:
    allreduce       — TP / EP 后 reduce
    alltoall        — EP dispatch / combine
    allgather       — sequence parallel reverse
    reduce_scatter  — sequence parallel forward
    p2p             — PD disaggregation / pipeline parallel

CollectiveOp 进 plan 后被 CostRouter 跳过 (stage 1-3 行为). Stage 5 引入
CommunicationFormulaBackend 才真正算 latency. 当前 formula 字段已对齐
V3 §5.4 OperatorSignature contract.
"""
from __future__ import annotations

from llm_infer_sim.core.operators.factories.common import make_runtime
from llm_infer_sim.core.operators.ops import CollectiveOp
from llm_infer_sim.core.operators.specs import OperatorFormula
from llm_infer_sim.core.profiles.deploy import DeployConfig


class CollectiveOpFactory:
    """生成 CollectiveOp. 不算 latency (留给 Stage 5 backend)."""

    def __init__(
        self,
        deploy: DeployConfig,
        *,
        backend: str = "nccl",
        topology: str = "single_node",
    ):
        self.deploy = deploy
        self.backend = backend
        self.topology = topology

    def allreduce(
        self,
        *,
        name: str,
        message_bytes: int,
        phase: str,
        layer_idx: int | None,
        world_size: int,
        dtype: str = "bf16",
    ) -> CollectiveOp:
        return self._build(
            op_subtype="allreduce", comm_type="allreduce",
            name=name, message_bytes=message_bytes,
            phase=phase, layer_idx=layer_idx,
            world_size=world_size, dtype=dtype,
        )

    def alltoall(
        self,
        *,
        name: str,
        message_bytes: int,
        phase: str,
        layer_idx: int | None,
        world_size: int,
        dtype: str = "bf16",
    ) -> CollectiveOp:
        return self._build(
            op_subtype="alltoall", comm_type="alltoall",
            name=name, message_bytes=message_bytes,
            phase=phase, layer_idx=layer_idx,
            world_size=world_size, dtype=dtype,
        )

    def allgather(
        self,
        *,
        name: str,
        message_bytes: int,
        phase: str,
        layer_idx: int | None,
        world_size: int,
        dtype: str = "bf16",
    ) -> CollectiveOp:
        return self._build(
            op_subtype="allgather", comm_type="allgather",
            name=name, message_bytes=message_bytes,
            phase=phase, layer_idx=layer_idx,
            world_size=world_size, dtype=dtype,
        )

    def reduce_scatter(
        self,
        *,
        name: str,
        message_bytes: int,
        phase: str,
        layer_idx: int | None,
        world_size: int,
        dtype: str = "bf16",
    ) -> CollectiveOp:
        return self._build(
            op_subtype="reduce_scatter", comm_type="reduce_scatter",
            name=name, message_bytes=message_bytes,
            phase=phase, layer_idx=layer_idx,
            world_size=world_size, dtype=dtype,
        )

    def p2p(
        self,
        *,
        name: str,
        message_bytes: int,
        phase: str,
        layer_idx: int | None,
        world_size: int = 2,
        dtype: str = "bf16",
    ) -> CollectiveOp:
        """Point-to-point send/recv (PD disaggregation, pipeline parallel)."""
        return self._build(
            op_subtype="p2p", comm_type="p2p",
            name=name, message_bytes=message_bytes,
            phase=phase, layer_idx=layer_idx,
            world_size=world_size, dtype=dtype,
        )

    # ---- shared builder ----

    def _build(
        self,
        *,
        op_subtype: str,
        comm_type: str,
        name: str,
        message_bytes: int,
        phase: str,
        layer_idx: int | None,
        world_size: int,
        dtype: str,
    ) -> CollectiveOp:
        runtime = {
            **make_runtime(self.deploy),
            "backend": self.backend,
            "topology": self.topology,
        }
        parallel = {
            "world_size": world_size,
            "tp": self.deploy.tp_size,
            "ep": self.deploy.ep_size,
        }
        formula = OperatorFormula(
            comm_bytes=float(message_bytes),
            comm_type=comm_type,
            op_category="communication",
        )
        return CollectiveOp(
            name=name, op_kind="collective", op_subtype=op_subtype,
            phase=phase, layer_idx=layer_idx, dtype=dtype,
            shape_fields={"message_bytes": int(message_bytes)},
            parallel_fields=parallel,
            runtime_fields=runtime,
            formula_value=formula,
        )
