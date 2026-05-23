"""Layer partition helper — grouped trace mode 用 (Task #154/#155).

把 model 的 num_layers 按 "同 op composition pattern" 分桶, 一个桶一个 GroupedOperator
代表. 当前 partition 规则:

FFN (partition_ffn_layers):
    Dense model (is_moe=False) → 1 桶 (all dense layers).
    MoE model:
        is_moe_layer(i) 为 False → "dense" FFN 桶
        其余 is_moe_layer(i) 为 True → "moe" 桶

attention 在 V3 / V3.2 / Qwen 上是 layer-uniform, 不需要 partition,
caller 直接用 num_layers 整体 group.

V4 hash MoE 分桶 / V4 attention 按 compress_ratio 分桶 已在 #157 删除,
未来基于新 op class 重新设计.
"""
from __future__ import annotations

from llm_infer_sim.core.profiles.model_config import ModelConfig


def partition_ffn_layers(model: ModelConfig) -> list[tuple[str, tuple[int, ...]]]:
    """返回 [(ffn_kind, layer_indices), ...] 顺序: dense → moe.

    ffn_kind ∈ {"dense", "moe"}.
    """
    if not model.is_moe:
        return [("dense", tuple(range(model.num_layers)))]

    dense_layers: list[int] = []
    moe_layers: list[int] = []
    for i in range(model.num_layers):
        if model.is_moe_layer(i):
            moe_layers.append(i)
        else:
            dense_layers.append(i)
    out: list[tuple[str, tuple[int, ...]]] = []
    if dense_layers:
        out.append(("dense", tuple(dense_layers)))
    if moe_layers:
        out.append(("moe", tuple(moe_layers)))
    return out
