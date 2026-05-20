"""StepCostEngine — IMPL_PLAN §1.4 Step 1.10.

最小闭环入口:
    workload -> StepShape -> ModelGraphTemplate -> StepOpPlan -> CostRouter -> StepCostTrace

阶段 1 只接 Qwen dense + RooflineBackend. 后续阶段把 template / router / backends 换成
更高优先级实现 (OperatorDB / ModuleProfile / etc.), 接口不变.
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.router import CostRouter
from llm_infer_sim.core.cost.trace import StepCostTrace
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.models.qwen import QwenModelGraphTemplate
from llm_infer_sim.core.ops.factories import (
    AttentionOpFactory,
    DenseOpFactory,
    EmbeddingOpFactory,
    FactoryBundle,
    NormalizationOpFactory,
)
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import GlobalStepWorkload


@dataclass(frozen=True)
class StepCostEngine:
    template: QwenModelGraphTemplate
    factories: FactoryBundle
    deploy: DeployConfig
    router: CostRouter

    def estimate(self, workload: GlobalStepWorkload) -> StepCostTrace:
        step = StepShape.from_workload(workload, self.deploy)
        plan = self.template.build_step(step, self.factories)
        return self.router.estimate(plan)


def build_qwen_dense_roofline_engine(
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
) -> StepCostEngine:
    """阶段 1 默认 Qwen dense + Roofline-only 装配."""
    factories = FactoryBundle(
        dense=DenseOpFactory(model, deploy),
        norm=NormalizationOpFactory(model, deploy),
        embedding=EmbeddingOpFactory(model, deploy),
        attention=AttentionOpFactory(model, deploy, hw),
    )
    backend = RooflineBackend(hw, deploy)
    router = CostRouter(backend)
    return StepCostEngine(
        template=QwenModelGraphTemplate(model),
        factories=factories,
        deploy=deploy,
        router=router,
    )
