"""Model-graph registry — 装饰器自注册 + 自动发现 (参考 aiconfigurator / vLLM).

每个模型图文件用 ``@register_model("<HF architectures[0]>")`` 注册一个工厂函数;
``models/__init__.py`` 用 ``pkgutil`` 自动 import 所有模型文件触发注册。**加新模型 = 加一个
文件 + 一行装饰器**, 不用改 registry / engine / __init__。

``get_model`` 按 ``ModelProfile.arch`` (= HF architecture 字符串, 跟 vLLM 模型
注册同口径) 精确字典查 —— 精确 key 匹配, 无优先级/顺序问题 (DeepSeek 同时是 MLA+MoE
也不冲突)。``arch`` 未设的手搭 profile (测试/调试) 按结构 (kv_lora_rank / num_experts)
兜底推断 arch。flat ``ModelConfig`` 经扁平 property facade duck-type 等价, 仍可传入。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from llm_infer_sim.core.models.base import BaseModel
    from llm_infer_sim.core.operators.context import OperatorContext
    from llm_infer_sim.core.models.config import ModelProfile

# arch 字符串 → 工厂函数 (model, ctx) -> BaseModel (模型图).
_REGISTRY: dict[str, "Callable[..., BaseModel]"] = {}

_AUTOLOADED = False


def _autoload_models() -> None:
    """首次 get_model 时 pkgutil import 所有模型图文件触发 @register_model 注册.

    推迟到调用时 (而非 package __init__) 以打破 operators.context →
    core.models.config → __init__ → 模型图 → operators.context 的循环 (见
    core/models/__init__.py docstring). registry / base / config / quantization
    / adapters 不是模型图文件, 跳过.
    """
    global _AUTOLOADED
    if _AUTOLOADED:
        return
    _AUTOLOADED = True
    import importlib
    import pkgutil

    from llm_infer_sim.core import models as _pkg

    _skip = ("registry", "base", "config", "quantization", "adapters")
    for _, _name, _ in pkgutil.iter_modules(_pkg.__path__):
        if _name not in _skip:
            importlib.import_module(f"{_pkg.__name__}.{_name}")


def register_model(*arch_names: str) -> Callable:
    """装饰器: 把工厂函数注册到一个或多个 HF architecture 字符串下."""
    def deco(factory: Callable) -> Callable:
        for arch in arch_names:
            _REGISTRY[arch] = factory
        return factory
    return deco


def _infer_arch(model: "ModelProfile") -> str:
    """arch 未设 (手搭 profile) 时按结构兜底推断 arch 字符串.

    生产路径 profile_extractor 会从 hf.architectures 设 model.arch, 走精确字典查;
    这里只服务测试/调试里手搭的 profile (它们关心的是结构, 不是具体 HF 模型)."""
    if model.kv_lora_rank > 0:
        return "DeepseekV3ForCausalLM"
    if model.num_experts > 0:
        return "Qwen3MoeForCausalLM"
    return "Qwen3ForCausalLM"


def get_model(
    model: "ModelProfile",
    ctx: "OperatorContext",
) -> "BaseModel":
    """按 arch 字典查工厂构造模型图; arch 未注册/未设时兜底推断."""
    _autoload_models()
    arch = model.arch or _infer_arch(model)
    factory = _REGISTRY.get(arch)
    if factory is None:
        raise ValueError(
            f"no model graph registered for arch={arch!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    return factory(model, ctx)
