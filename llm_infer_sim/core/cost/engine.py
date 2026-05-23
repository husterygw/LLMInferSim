"""StepCostEngine — IMPL_PLAN §1.4 Step 1.10.

最小闭环入口:
    workload -> StepShape -> ModelGraphTemplate -> GroupedStepPlan -> CostRouter -> StepCostTrace

Template 持 OperatorContext, 直接构造 op classes (无中间 factory 层).
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.router import CostRouter
from llm_infer_sim.core.cost.trace import StepCostTrace
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.models.deepseek import DeepSeekModelTemplate
from llm_infer_sim.core.models.qwen import QwenModelGraphTemplate
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.operators import MoERoutingProfile
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.efficiency_profile import EfficiencyProfile
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import GlobalStepWorkload


@dataclass(frozen=True)
class StepCostEngine:
    template: QwenModelGraphTemplate | DeepSeekModelTemplate
    deploy: DeployConfig
    router: CostRouter

    def estimate(self, workload: GlobalStepWorkload) -> StepCostTrace:
        step = StepShape.from_workload(workload, self.deploy)
        grouped = self.template.build_grouped_step(step)
        return self.router.estimate_grouped(grouped)


def build_qwen_dense_roofline_engine(
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    *,
    efficiency: EfficiencyProfile | None = None,
) -> StepCostEngine:
    """阶段 1 默认 Qwen dense + Roofline-only 装配."""
    ctx = build_operator_context(model, deploy, hw, efficiency)
    return StepCostEngine(
        template=QwenModelGraphTemplate(model=model, ctx=ctx),
        deploy=deploy,
        router=CostRouter(RooflineBackend(hw, deploy)),
    )


def build_deepseek_roofline_engine(
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    *,
    routing: MoERoutingProfile | None = None,
    indexer_kv_byte: float = 1.0,
    efficiency: EfficiencyProfile | None = None,
) -> StepCostEngine:
    """阶段 3b/3c: DeepSeek V3 dense MLA / V3.2 sparse MLA + dense/MoE FFN."""
    ctx = build_operator_context(model, deploy, hw, efficiency)
    return StepCostEngine(
        template=DeepSeekModelTemplate(
            model=model, ctx=ctx,
            routing=routing, indexer_kv_byte=indexer_kv_byte,
        ),
        deploy=deploy,
        router=CostRouter(RooflineBackend(hw, deploy)),
    )


def build_qwen_roofline_engine(
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    *,
    routing: MoERoutingProfile | None = None,
    efficiency: EfficiencyProfile | None = None,
) -> StepCostEngine:
    """阶段 3a 起的通用 Qwen 装配 (含 MoE / collective)."""
    ctx = build_operator_context(model, deploy, hw, efficiency)
    return StepCostEngine(
        template=QwenModelGraphTemplate(model=model, ctx=ctx, routing=routing),
        deploy=deploy,
        router=CostRouter(RooflineBackend(hw, deploy)),
    )
