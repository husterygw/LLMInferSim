"""Factory 共享 helper — OperatorProfile → VirtualOp.formula 转换 + runtime/parallel context."""
from __future__ import annotations

from llm_infer_sim.core.ops.base import OperatorProfile
from llm_infer_sim.core.profiles.deploy import DeployConfig


def profile_to_formula(op: OperatorProfile, op_precision: str = "bf16") -> dict:
    """OperatorProfile -> VirtualOp.formula (V3 §4.3 keys)."""
    return {
        "flops": op.flops,
        "load_weight": op.load_weight,
        "load_act": op.load_act,
        "store_act": op.store_act,
        "load_kv_cache": op.load_kv_cache,
        "store_kv_cache": op.store_kv_cache,
        "op_precision": op_precision,
        "comm_bytes": op.comm_bytes,
        "comm_type": op.comm_type,
        "op_category": op.op_category,
    }


def make_runtime(deploy: DeployConfig, *, kernel_source: str = "vllm_default") -> dict:
    return {
        "framework": deploy.backend,
        "framework_version": deploy.backend_version or "unknown",
        "execution_mode": deploy.execution_mode,
        "kernel_source": kernel_source,
    }


def dense_parallel(deploy: DeployConfig) -> dict:
    return {"tp": deploy.tp_size}
