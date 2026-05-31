"""BaseModel — 模型图的结构化协议 (op_plan §7).

模型图 (Qwen3Model / Qwen3MoeModel / DeepSeekModel / 未来新增) 只需实现
``forward(step) -> StepOpPlan`` 即满足此协议; 不要求显式继承 (structural typing,
跟 operators/base.py 的 Operator 协议同套路)。StepCostEngine / registry 用它做类型,
**加新模型不用改 engine / registry 的类型签名**。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm_infer_sim.core.step.step_plan import StepOpPlan
from llm_infer_sim.core.step.step_shape import StepShape


@runtime_checkable
class BaseModel(Protocol):
    """A model graph: forward(step) → StepOpPlan (op list + step runtime)."""

    def forward(self, step: StepShape) -> StepOpPlan: ...
