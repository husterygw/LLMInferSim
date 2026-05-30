"""Model-graph classes — op_plan §7 (build-once).

模型图通过 ``@register_model(...)`` 自注册 (registry.py)。注册触发改为**惰性**:
``registry.get_model()`` 首次调用时 pkgutil 自动 import 所有模型图文件 (见
``registry._autoload_models``)。

惰性化原因 (config_plan Step 5): flat ``ModelConfig`` 已迁入本包 (config.py),
operators 层会 ``from core.models.config import ModelConfig`` —— 若 __init__ 在
package load 时就 eager import 模型图 (deepseek/qwen3, 它们 import operators.context),
会与 operators.context → core.models.config 形成循环 import。把模型图 import 推迟到
get_model 调用时即可解环 (那时各层均已加载完毕)。
"""
from __future__ import annotations

from llm_infer_sim.core.models.base import BaseModel
from llm_infer_sim.core.models.registry import get_model, register_model

__all__ = [
    "BaseModel",
    "register_model",
    "get_model",
]
