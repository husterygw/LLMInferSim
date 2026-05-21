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
from llm_infer_sim.core.models.deepseek import DeepSeekModelTemplate
from llm_infer_sim.core.models.qwen import QwenModelGraphTemplate
from llm_infer_sim.core.operators.factories import (
    AttentionOpFactory,
    CollectiveOpFactory,
    DenseOpFactory,
    EmbeddingOpFactory,
    FactoryBundle,
    IndexerOpFactory,
    MoEOpFactory,
    NormalizationOpFactory,
    V4AttentionOpFactory,
)
from llm_infer_sim.core.operators.routing import MoERoutingProfile
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import GlobalStepWorkload


@dataclass(frozen=True)
class StepCostEngine:
    template: QwenModelGraphTemplate | DeepSeekModelTemplate
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


def build_deepseek_roofline_engine(
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    *,
    routing: MoERoutingProfile | None = None,
    indexer_kv_byte: float = 1.0,
) -> StepCostEngine:
    """阶段 3b/3c: DeepSeek V3 dense MLA / V3.2 sparse MLA + dense/MoE FFN.

    template 自动按 ``model.index_topk > 0`` 判断走 V3.2 sparse path.
    """
    indexer = IndexerOpFactory(model, deploy, indexer_kv_byte=indexer_kv_byte)
    factories = FactoryBundle(
        dense=DenseOpFactory(model, deploy),
        norm=NormalizationOpFactory(model, deploy),
        embedding=EmbeddingOpFactory(model, deploy),
        attention=AttentionOpFactory(model, deploy, hw),
        moe=MoEOpFactory(model, deploy, routing=routing),
        collective=CollectiveOpFactory(deploy),
        indexer=indexer,
        v4_attention=V4AttentionOpFactory(model, deploy, hw),
    )
    backend = RooflineBackend(hw, deploy)
    router = CostRouter(backend)
    return StepCostEngine(
        template=DeepSeekModelTemplate(model),
        factories=factories,
        deploy=deploy,
        router=router,
    )


def build_qwen_roofline_engine(
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    *,
    routing: MoERoutingProfile | None = None,
) -> StepCostEngine:
    """阶段 3a 起的通用 Qwen 装配 (含 MoE / collective).

    model.is_moe=False 时退化到 dense-only 行为, 不影响 阶段 1 路径.
    """
    factories = FactoryBundle(
        dense=DenseOpFactory(model, deploy),
        norm=NormalizationOpFactory(model, deploy),
        embedding=EmbeddingOpFactory(model, deploy),
        attention=AttentionOpFactory(model, deploy, hw),
        moe=MoEOpFactory(model, deploy, routing=routing),
        collective=CollectiveOpFactory(deploy),
    )
    backend = RooflineBackend(hw, deploy)
    router = CostRouter(backend)
    return StepCostEngine(
        template=QwenModelGraphTemplate(model),
        factories=factories,
        deploy=deploy,
        router=router,
    )
