"""Model family adapter dispatcher.

每个 family 一个适配器模块, 提供 6 个 getter:
  get_num_attention_heads / get_num_key_value_heads / get_hidden_size /
  get_num_hidden_layers / get_intermediate_size / get_vocab_size

通过 hf_config.model_type (vllm 已解析) dispatch。

阶段 2 范围: 只注册 opt + qwen 两家。
后续阶段:
  - Llama 系列在阶段 4 切到大 dense 模型时加
  - DeepSeek-V3 / V4 在阶段 8/9 加
"""
from __future__ import annotations

from llm_infer_sim.core.profiles.model_adapters import deepseek_v3 as _deepseek_v3
from llm_infer_sim.core.profiles.model_adapters import deepseek_v4 as _deepseek_v4
from llm_infer_sim.core.profiles.model_adapters import opt as _opt
from llm_infer_sim.core.profiles.model_adapters import qwen as _qwen


# hf_config.model_type → adapter module
ADAPTERS: dict[str, object] = {
    "opt": _opt,
    "qwen2": _qwen,
    "qwen2_moe": _qwen,
    "qwen3": _qwen,
    "qwen3_moe": _qwen,
    "deepseek_v3": _deepseek_v3,
    "deepseek_v32": _deepseek_v3,    # V3.2-Exp 字段基础同 V3, indexer 字段独立透传
    "deepseek_v4": _deepseek_v4,
}


class UnsupportedModelError(ValueError):
    """vLLM hf_config.model_type 不在 ADAPTERS 里。"""


def get_adapter(model_type: str):
    if model_type not in ADAPTERS:
        raise UnsupportedModelError(
            f"model_type={model_type!r} 不在阶段 2 已支持适配器列表 "
            f"{list(ADAPTERS)} 中。新增模型请先添加 model_adapters/<family>.py。"
        )
    return ADAPTERS[model_type]
