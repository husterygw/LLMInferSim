"""Model family adapter dispatcher (config_plan Step 5: 从 core/profiles/model_adapters 迁入)。

每个 family 一个适配器模块, 提供 6 个 getter:
  get_num_attention_heads / get_num_key_value_heads / get_hidden_size /
  get_num_hidden_layers / get_intermediate_size / get_vocab_size

通过 hf_config.model_type (vllm 已解析) dispatch。
"""
from __future__ import annotations

from llm_infer_sim.core.models.adapters import deepseek_v3 as _deepseek_v3
from llm_infer_sim.core.models.adapters import deepseek_v4 as _deepseek_v4
from llm_infer_sim.core.models.adapters import opt as _opt
from llm_infer_sim.core.models.adapters import qwen as _qwen


# hf_config.model_type → adapter module
ADAPTERS: dict[str, object] = {
    "opt": _opt,
    "qwen2": _qwen,
    "qwen2_moe": _qwen,
    "qwen3": _qwen,
    "qwen3_moe": _qwen,
    "deepseek_v3": _deepseek_v3,
    "deepseek_v4": _deepseek_v4,
}


class UnsupportedModelError(ValueError):
    """vLLM hf_config.model_type 不在 ADAPTERS 里。"""


def get_adapter(model_type: str):
    if model_type not in ADAPTERS:
        raise UnsupportedModelError(
            f"model_type={model_type!r} 不在已支持适配器列表 "
            f"{list(ADAPTERS)} 中。新增模型请先添加 models/adapters/<family>.py。"
        )
    return ADAPTERS[model_type]
