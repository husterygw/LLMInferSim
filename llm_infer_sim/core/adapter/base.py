"""IFrameworkAdapter — 推理框架适配层抽象接口定义。

各框架适配器负责三件事:
  1. 把框架的 step 调度上下文转换为 GlobalStepWorkload (框架无关)
  2. 向框架注入虚拟执行后端 (接管 model execution)
  3. 把 GlobalStepCost 翻译为框架期望的输出格式 (fake output)

阶段 0 spike: 建好接口契约, 暂无具体实现者 (vLLM 路径直接走
VirtualPlatform → VirtualWorker, 不走 VllmAdapter)。后续阶段 (尤其
SGLang adapter) 才落实施。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_infer_sim.core.workload.workload import GlobalStepWorkload
from llm_infer_sim.core.cost_model.cost_result import GlobalStepCost


class IFrameworkAdapter(ABC):
    """推理框架适配层抽象基类。

    同一套 Core 仿真引擎可通过不同 Adapter 接入不同框架:

        EngineCore (框架自身)
            |
            v
        IFrameworkAdapter.get_workload(step_ctx)        # Adapter 转换
            |
            v
        Core Engine: plan → cost → simulate
            |
            v
        IFrameworkAdapter.emit_result(workload, result) # Adapter 反向转换
    """

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """框架标识字符串, e.g. 'vllm', 'sglang'."""
        ...

    @abstractmethod
    def register_backend(self, framework_config: Any) -> None:
        """向推理框架注入虚拟执行后端 (在框架初始化期间调用)。

        - vLLM: 通过 entry_point 注册 OOT Platform; 或设 VLLM_VIRTUAL_BACKEND=1
        - SGLang: hook ServerArgs.model_executor_cls (待调研)
        """
        ...

    @abstractmethod
    def get_workload(self, step_context: Any) -> GlobalStepWorkload:
        """把框架的 step 调度上下文翻译为框架无关的 GlobalStepWorkload。

        step_context 类型依框架而定:
          vLLM   : SchedulerOutput
          SGLang : ScheduleBatch

        实现约束:
          - 输出必须含 num_prefill_tokens / num_decode_tokens / requests
          - 不得含框架特定对象引用
          - chunked prefill: phase=CHUNKED_PREFILL, is_chunked=True
        """
        ...

    @abstractmethod
    def emit_result(
        self, workload: GlobalStepWorkload, result: GlobalStepCost
    ) -> Any:
        """把 GlobalStepCost 翻译为框架期望的输出格式并返回。

        - vLLM   : 返 ModelRunnerOutput (req_ids, sampled_token_ids, ...)
        - SGLang : 返 BatchTokenIdOut
        """
        ...

    # ---- 可选 hook ----

    def on_request_finished(self, request_id: str, metrics: dict) -> None:
        """框架请求完成时通知 adapter (可选)。"""
        return None

    def get_model_config(self) -> dict:
        """把框架的模型配置转成 perf_sim 可读形式 (可选)。"""
        return {}
