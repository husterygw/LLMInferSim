"""OperatorContext + ModelBuildContext — Op class refactor 配置上下文 (#158).

将 model / deploy / hw / dtype / byte 字段一次性绑定, 让 op class 不再重复传 byte / runtime.

- OperatorContext: 每个 op 持有, immutable. 包含 op-level formula 所需的所有共享配置.
  签名 (compare/hash) 时设 compare=False, 不污染 OperatorDB lookup.
- ModelBuildContext: 模型模板持有, 包含 OperatorContext + 模板构图时才用到的 routing /
  indexer_kv_byte 等. op 实例只 carry OperatorContext, 不 carry routing.

build_operator_context(model, deploy, hw, efficiency=None) 从配置一次推导出 ctx,
在 engine builder 里调用.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.efficiency_profile import EfficiencyProfile
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig

if TYPE_CHECKING:
    from llm_infer_sim.core.operators.moe import MoERoutingProfile


@dataclass(frozen=True)
class OperatorContext:
    """Op 共享配置 (model / deploy / hw + byte / dtype).

    每个 op 实例持有一个 ctx 引用; 多 op 共享同一 ctx (immutable, 安全).
    roofline_spec() / shape / parallel / runtime 从这里读 byte / framework / execution_mode.

    OperatorDB signature 不含 ctx 内容 (byte 字段是 roofline 公式输入, 不进 DB key).
    op 类用 dataclass field(compare=False) 把 ctx 排除在 hash/eq 之外.
    """
    model: ModelConfig
    deploy: DeployConfig
    hw: HardwareConfig
    w_byte: float = 2.0      # 量化层 dtype (bf16=2.0, fp8=1.0, fp4=0.5)
    a_byte: float = 2.0      # activation dtype
    kv_byte: float = 2.0     # KV cache dtype
    dtype: str = "bf16"      # op default precision; per-op override 通过构造参数

    # ---- runtime convenience (避免每个 op formula 都 self.ctx.deploy.xxx) ----

    @property
    def framework(self) -> str:
        return self.deploy.backend

    @property
    def framework_version(self) -> str:
        return self.deploy.backend_version or "unknown"

    @property
    def execution_mode(self) -> str:
        return self.deploy.execution_mode

    @property
    def tp_size(self) -> int:
        return self.deploy.tp_size

    @property
    def ep_size(self) -> int:
        return self.deploy.ep_size

    @property
    def block_size(self) -> int:
        return self.deploy.block_size


@dataclass(frozen=True)
class ModelBuildContext:
    """模板构图上下文 (OperatorContext + routing / indexer 配置).

    template.build_grouped_step(step, mbc) 从 mbc.op 派生 op, 从 mbc.routing 算
    MoE expert count / alltoall message bytes.
    """
    op: OperatorContext
    routing: "MoERoutingProfile | None" = None
    indexer_kv_byte: float = 1.0      # V3.2 indexer KV cache 一般 fp8 (1.0 byte/elem)


def build_operator_context(
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    efficiency: EfficiencyProfile | None = None,
) -> OperatorContext:
    """从配置一次性推导 OperatorContext.

    efficiency=None 时用 placeholder (全 1.0 / bf16). 实际部署由 adapter
    (profile_extractor) 从 quantization_config 推导 EfficiencyProfile 后传入.
    """
    eff = efficiency or EfficiencyProfile.placeholder()
    return OperatorContext(
        model=model,
        deploy=deploy,
        hw=hw,
        w_byte=eff.w_byte,
        a_byte=eff.a_byte,
        kv_byte=eff.kv_byte,
        dtype="bf16",   # 默认 bf16; 未来由 efficiency 推导
    )


def build_model_build_context(
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    *,
    efficiency: EfficiencyProfile | None = None,
    routing: "MoERoutingProfile | None" = None,
    indexer_kv_byte: float = 1.0,
) -> ModelBuildContext:
    """One-shot builder: 配置 → ModelBuildContext."""
    op_ctx = build_operator_context(model, deploy, hw, efficiency)
    return ModelBuildContext(
        op=op_ctx,
        routing=routing,
        indexer_kv_byte=indexer_kv_byte,
    )
