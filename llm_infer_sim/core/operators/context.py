"""OperatorContext — Op class refactor 配置上下文 (#158).

将 model / deployment / runtime / hw / dtype / byte / routing 字段一次性绑定, 让 op
class 不再重复传 byte / runtime. 每个 op 持有一个 ctx 引用 (immutable); op 类用
field(compare=False) 把 ctx 排除在 hash/eq 之外, 不污染 OperatorDB lookup.

ctx 持结构化 DeploymentProfile + RuntimeProfile; op formula 经 property
(framework / execution_mode / tp_size / ...) 读, 不再持 flat deploy 对象。

build_operator_context(model, deployment, runtime, hw, ...) 是唯一装配入口
(scenario / 测试共用)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from typing import TYPE_CHECKING

from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.models.config import ModelProfile
from llm_infer_sim.core.hardware.device import HardwareProfile
from llm_infer_sim.core.models.quantization import QuantizationProfile

if TYPE_CHECKING:
    from llm_infer_sim.core.operators.moe import MoERoutingProfile
    from llm_infer_sim.core.runtime.profile import RuntimeProfile
    from llm_infer_sim.core.scenario import SimulationScenario


@dataclass(frozen=True)
class OperatorContext:
    """Op 共享配置 (model / deployment / runtime / hw + byte / dtype).

    每个 op 实例持有一个 ctx 引用; 多 op 共享同一 ctx (immutable, 安全).
    roofline_spec() / shape / parallel / runtime 从这里读 byte / framework / execution_mode.

    OperatorDB signature 不含 ctx 内容 (byte 字段是 roofline 公式输入, 不进 DB key).
    op 类用 dataclass field(compare=False) 把 ctx 排除在 hash/eq 之外.
    """
    model: ModelProfile
    deployment: DeploymentProfile
    runtime: RuntimeProfile
    hw: HardwareProfile
    w_byte: float = 2.0      # 量化层 dtype (bf16=2.0, fp8=1.0, fp4=0.5)
    a_byte: float = 2.0      # activation dtype
    kv_byte: float = 2.0     # KV cache dtype
    dtype: str = "bf16"      # op default precision; per-op override 通过构造参数
    # MoE expert-routing 假设 (skew → distinct experts → routed_experts weight read).
    # 是部署级成本假设, 不是结构身份 → compare=False (不进 hash/eq, 不污染 DB key).
    # 模型图建图时读 ctx.routing; dense 模型忽略它. None → 各 MoE 模型默认 balanced().
    routing: "MoERoutingProfile | None" = field(default=None, compare=False)

    # ---- runtime convenience (避免每个 op formula 都 self.ctx.<...>) ----

    @property
    def framework(self) -> str:
        return self.runtime.framework.name

    @property
    def framework_version(self) -> str:
        return self.runtime.framework.version or "unknown"

    @property
    def execution_mode(self) -> str:
        return self.runtime.execution.execution_mode

    @property
    def tp_size(self) -> int:
        return self.deployment.parallelism.tp

    @property
    def ep_size(self) -> int:
        return self.deployment.parallelism.ep

    @property
    def block_size(self) -> int:
        return self.deployment.kv_cache.block_size


def build_operator_context(
    model: ModelProfile,
    deployment: DeploymentProfile,
    runtime: "RuntimeProfile",
    hw: HardwareProfile,
    *,
    quantization: QuantizationProfile | None = None,
    routing: "MoERoutingProfile | None" = None,
) -> OperatorContext:
    """从结构化域对象一次性推导 OperatorContext (唯一装配入口)。

    quantization=None 时用 placeholder (bf16 全 2.0). routing=None → MoE 模型图建图时
    退回 balanced().
    """
    quant = quantization or QuantizationProfile.placeholder()
    return OperatorContext(
        model=model,
        deployment=deployment,
        runtime=runtime,
        hw=hw,
        w_byte=quant.w_byte,
        a_byte=quant.a_byte,
        kv_byte=quant.kv_byte,
        dtype="bf16",   # 默认 bf16; 未来由 quantization 推导
        routing=routing,
    )


def build_operator_context_from_scenario(
    scenario: "SimulationScenario",
) -> OperatorContext:
    """从 SimulationScenario 装配 OperatorContext (生产唯一入口)。

    把 scenario → 域对象的拆包逻辑收敛在此, 调用方只持 scenario。
    """
    return build_operator_context(
        scenario.model,
        scenario.deployment,
        scenario.runtime,
        scenario.hardware,
        quantization=scenario.model.quantization,
        routing=scenario.runtime.kernels.moe_routing,
    )
