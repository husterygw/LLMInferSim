"""RuntimeProfile — runtime 域聚合 (config_plan §4.5)。

framework / execution mode / cuda graph / kernel policy / 通信实现选择。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_infer_sim.core.runtime.communication import RuntimeCommunicationPolicy
from llm_infer_sim.core.runtime.execution import ExecutionProfile
from llm_infer_sim.core.runtime.framework import FrameworkProfile
from llm_infer_sim.core.runtime.graph import CudaGraphProfile
from llm_infer_sim.core.runtime.kernels import KernelBackendProfile


@dataclass(frozen=True)
class RuntimeProfile:
    framework: FrameworkProfile = field(default_factory=FrameworkProfile)
    execution: ExecutionProfile = field(default_factory=ExecutionProfile)
    graph: CudaGraphProfile | None = None
    kernels: KernelBackendProfile = field(default_factory=KernelBackendProfile)
    communication: RuntimeCommunicationPolicy = field(
        default_factory=RuntimeCommunicationPolicy
    )

    @classmethod
    def flat(
        cls,
        *,
        execution_mode: str = "eager",
        backend: str = "vllm",
        backend_version: str | None = None,
        prefill_worker_overhead_s: float = 0.005,
        kernel_profile: KernelBackendProfile | None = None,
    ) -> "RuntimeProfile":
        """扁平 kwargs 构造 (测试 / adapter 便捷入口)。

        kernel_profile 提供 kernel policy (None → KernelBackendProfile 默认)。
        graph 由 execution_mode 派生 (numerically inert, 仅记录)。
        """
        return cls(
            framework=FrameworkProfile(name=backend, version=backend_version),
            execution=ExecutionProfile(
                execution_mode=execution_mode,
                prefill_worker_overhead_s=prefill_worker_overhead_s,
            ),
            graph=CudaGraphProfile(enabled=execution_mode == "cudagraph"),
            kernels=kernel_profile or KernelBackendProfile(),
            communication=RuntimeCommunicationPolicy(),
        )
