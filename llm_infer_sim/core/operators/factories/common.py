"""Shared helpers for Operator factories."""
from __future__ import annotations

from llm_infer_sim.core.profiles.deploy import DeployConfig


def make_runtime(deploy: DeployConfig, *, kernel_source: str = "vllm_default") -> dict:
    return {
        "framework": deploy.backend,
        "framework_version": deploy.backend_version or "unknown",
        "execution_mode": deploy.execution_mode,
        "kernel_source": kernel_source,
    }


def dense_parallel(deploy: DeployConfig) -> dict:
    return {"tp": deploy.tp_size}
