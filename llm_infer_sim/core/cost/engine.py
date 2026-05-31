"""StepCostEngine — IMPL_PLAN §1.4 Step 1.10.

最小闭环入口:
    workload -> StepShape -> model.forward -> StepOpPlan -> CostRouter -> StepCostTrace

模型图 (BaseModel) 持 OperatorContext, 直接构造 op classes (无中间 factory 层).
engine 不依赖具体模型类 —— 装配统一走 registry.get_model (按 arch 分发)。
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.router import CostRouter
from llm_infer_sim.core.cost.trace import StepCostTrace
from llm_infer_sim.core.step.step_shape import StepShape
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.models.base import BaseModel
from llm_infer_sim.core.models.registry import get_model
from llm_infer_sim.core.operators.context import (
    build_operator_context,
    build_operator_context_from_scenario,
)
from llm_infer_sim.core.operators import MoERoutingProfile
from llm_infer_sim.core.calibration.profile import CalibrationProfile
from llm_infer_sim.core.hardware.device import HardwareConfig
from llm_infer_sim.core.models.config import ModelConfig
from llm_infer_sim.core.models.quantization import QuantizationProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.scenario import SimulationScenario
from llm_infer_sim.core.workload.workload import GlobalStepWorkload


@dataclass(frozen=True)
class StepCostEngine:
    model: BaseModel
    execution_mode: str
    router: CostRouter

    def estimate(self, workload: GlobalStepWorkload) -> StepCostTrace:
        step = StepShape.from_workload(workload, self.execution_mode)
        # static-contract path (op_plan): model.forward() → StepOpPlan → router.
        return self.router.estimate(self.model.forward(step))


def build_roofline_engine(
    model: ModelConfig,
    deployment: DeploymentProfile,
    runtime: RuntimeProfile,
    hw: HardwareConfig,
    *,
    routing: MoERoutingProfile | None = None,
    quantization: QuantizationProfile | None = None,
    calibration: CalibrationProfile | None = None,
) -> StepCostEngine:
    """**Compatibility builder** — 只接收扁平 ModelConfig / HardwareConfig, 供测试 /
    历史 baseline / 兼容 builder 直传扁平域对象。

    生产代码不要新增对本函数的调用: 装配链应走结构化的
    build_roofline_engine_from_scenario(scenario)。新增 model/hardware 字段先加到
    structured profile, 再视需要补 flat facade (见 .claude/CLAUDE.md 工程约定)。
    """
    ctx = build_operator_context(
        model, deployment, runtime, hw, quantization=quantization, routing=routing,
    )
    execution_mode = runtime.execution.execution_mode
    return StepCostEngine(
        model=get_model(model, ctx),
        execution_mode=execution_mode,
        router=CostRouter(RooflineBackend(hw, execution_mode, calibration=calibration)),
    )


def build_roofline_engine_from_scenario(
    scenario: SimulationScenario,
) -> StepCostEngine:
    """从结构化 SimulationScenario 直接装配 (config_plan Step C)。

    不再 to_legacy() 折返: ModelProfile / HardwareProfile 经扁平 read facade 直接喂给
    OperatorContext / get_model / RooflineBackend, 数值与旧 legacy 路径 byte-identical。
    """
    ctx = build_operator_context_from_scenario(scenario)
    execution_mode = scenario.runtime.execution.execution_mode
    return StepCostEngine(
        model=get_model(scenario.model, ctx),
        execution_mode=execution_mode,
        router=CostRouter(
            RooflineBackend(
                scenario.hardware, execution_mode, calibration=scenario.calibration,
            )
        ),
    )


# 下面三个命名 builder 是 build_roofline_engine 的薄包装 (历史调用方/测试用); 都经
# get_model 按 arch 分发, engine 不直接构造具体模型类。
def build_qwen_dense_roofline_engine(
    model: ModelConfig,
    deployment: DeploymentProfile,
    runtime: RuntimeProfile,
    hw: HardwareConfig,
    *,
    quantization: QuantizationProfile | None = None,
    calibration: CalibrationProfile | None = None,
) -> StepCostEngine:
    return build_roofline_engine(
        model, deployment, runtime, hw,
        quantization=quantization, calibration=calibration,
    )


def build_qwen_roofline_engine(
    model: ModelConfig,
    deployment: DeploymentProfile,
    runtime: RuntimeProfile,
    hw: HardwareConfig,
    *,
    routing: MoERoutingProfile | None = None,
    quantization: QuantizationProfile | None = None,
    calibration: CalibrationProfile | None = None,
) -> StepCostEngine:
    return build_roofline_engine(
        model, deployment, runtime, hw,
        routing=routing, quantization=quantization, calibration=calibration,
    )


def build_deepseek_roofline_engine(
    model: ModelConfig,
    deployment: DeploymentProfile,
    runtime: RuntimeProfile,
    hw: HardwareConfig,
    *,
    routing: MoERoutingProfile | None = None,
    quantization: QuantizationProfile | None = None,
    calibration: CalibrationProfile | None = None,
) -> StepCostEngine:
    return build_roofline_engine(
        model, deployment, runtime, hw,
        routing=routing, quantization=quantization, calibration=calibration,
    )
