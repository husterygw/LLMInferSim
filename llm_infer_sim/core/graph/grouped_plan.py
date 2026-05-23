"""GroupedStepPlan — V3 §4.4 grouped trace mode 的 plan 表达.

dense decoder 一个 step 里同 layer pattern 的 op 形状/输入完全相同 (e.g.
Qwen3-4B 36 层 qkv_proj 都是 m=tokens, n=6144, k=2560). roofline 公式
latency 跟 layer_idx 无关, 所以可以只对一个代表 op 算一次, 然后乘 count.

GroupedStepPlan 在结构上显式表达这件事:
- groups: tuple[GroupedOperator], 每个 group 一个代表 op + count + layer_indices
- count > 1 → 同 pattern 的多 layer 合并
- count == 1 → 全局唯一 op (embedding / lm_head)

与 StepOpPlan 数学等价: total_latency = sum(latency_per_op × count) =
sum(per_layer_latency for all layers) = full StepOpPlan 的 total_latency.

设计文档: docs/IMPL_PLAN.md §grouped-trace + memory feedback_grouped_trace_kills_full_path.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.operators.base import Operator


@dataclass(frozen=True)
class GroupedOperator:
    """同形状 layer op 的合并表达.

    op: 代表 op 实例 (通常是 layer_indices[0] 对应的那个 op, 全部 layer 用相同 op).
    count: 等价 op 数量 (= len(layer_indices)).
    layer_indices: 这个 group 覆盖的具体 layer idx 列表 (供 debug / report).
    """
    op: Operator
    count: int
    layer_indices: tuple[int, ...]


@dataclass(frozen=True)
class GroupedStepPlan:
    step_id: int
    phase: str
    groups: tuple[GroupedOperator, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ops(self) -> tuple[Operator, ...]:
        """Flatten groups → ops tuple (test/debug helper).

        每个 GroupedOperator.op 按 count 复制. 顺序 = build_grouped_step 的 group 顺序
        (embedding → attention block ops → FFN block ops → lm_head), NOT 原始 per-layer
        interleave 顺序. 因此 ops[i:j] 不再是 "layer k 的连续 op", 但单 op 形状 / subtype /
        filter-by-predicate 用法不变.

        per-layer 视角请用: ``[g.op for g in plan.groups if layer_idx in g.layer_indices]``.
        """
        flat: list[Operator] = []
        for g in self.groups:
            flat.extend([g.op] * g.count)
        return tuple(flat)
