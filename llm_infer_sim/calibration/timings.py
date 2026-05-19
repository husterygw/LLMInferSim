"""LayerwiseProfileResults 树 → TimingSample 提取 (详设 §9.4.2 B.1).

vLLM 的 `layerwise_profile()` 在 __exit__ 后产出 `LayerwiseProfileResults`,
调它的 `convert_stats_to_dict()` 得到嵌套结构:

  {
    "metadata": {...},
    "summary_stats": [...],
    "model_stats": [               # ← 我们走这条 (per-module 嵌套树)
        {
            "entry": {"name": "Qwen3Model(...)", "cuda_time_us": 12345.6,
                      "invocations": 1, "cpu_time_us": ...},
            "children": [
                {"entry": {"name": "VocabParallelEmbedding(...)", ...},
                 "children": [...]},
                ...
            ]
        },
        ...
    ]
  }

我们做 DFS, 在每个 node:
  1. 剥 `ClassName(repr)` → 类名
  2. 用 catalog.match(node_class, ancestors) 匹 canonical
  3. 命中: per_invocation_us = cuda_time_us / invocations, 产生 TimingSample

跨 iterations 的均值由调用方 (extension.fire) 在 layerwise_profile 内重复 forward
N 次, vLLM 自动累加 cuda_time_us 和 invocations, 我们 / invocations 即得 per-call.

参考 LLMServingSim profiler/core/hooks/timings.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TimingSample:
    """单个 canonical 的一次测量 (per-invocation 均值)."""
    layer: str               # canonical name (跟 catalog.canonical 对应)
    op_kind: str             # 透传 catalog 的 op_kind, fit 时用
    microseconds: float      # per-invocation cuda time

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_class_name(raw: str) -> str:
    """`QKVParallelLinear(in_features=4096, ...)` → `QKVParallelLinear`."""
    paren = raw.find("(")
    return raw if paren < 0 else raw[:paren]


def extract_samples(
    model_stats: list[dict[str, Any]] | dict[str, Any],
    catalog_slice: dict[str, dict[str, Any]],
) -> list[TimingSample]:
    """DFS 走 model_stats 树, 匹 catalog_slice, 返回 TimingSample 列表.

    Args:
        model_stats: 来自 `LayerwiseProfileResults.convert_stats_to_dict()["model_stats"]`,
                     可能是 list (顶层多 root) 或 dict (single root). 我们兼容两种.
        catalog_slice: `{canonical: {"vllm": cls, "within": parent_or_None, "op_kind": ...}}`
                       由 catalog.slice_for_op_kinds() 产生.

    Returns:
        list[TimingSample]. 没匹中的 node 不出现; 同 canonical 多次匹中 (例 36 layers)
        各产一条 sample (不在这里聚合, fit 阶段做 median).
    """
    samples: list[TimingSample] = []
    roots = model_stats if isinstance(model_stats, list) else [model_stats]
    for root in roots:
        _dfs(root, ancestors=[], slice_=catalog_slice, out=samples)
    return samples


def _dfs(
    node: dict[str, Any],
    ancestors: list[str],
    slice_: dict[str, dict[str, Any]],
    out: list[TimingSample],
) -> None:
    """DFS 走 model_stats tree.

    vLLM `model_stats` entry schema (0.19.1 / 0.20.x 一致):
      {"name", "cpu_time_us", "cuda_time_us", "pct_cuda_time", "trace"}
    **不含 invocations**: model_stats 树展开了 per-call, 同 module 多次调用变兄弟节点
    (例 36 个 layer × QKVParallelLinear → 36 个独立子树 entry, 各自 cuda_us).
    所以这里每命中一次产 1 个 sample, **不除 invocations**; 跨调用聚合 (取 median 等)
    交给 fit 阶段。

    历史: 早期我们误以为字段含 invocations (那是 summary_stats 的字段, 走聚合视图);
    一直没匹中 sample 直到 cu12 实测 (B.4).
    """
    entry = node.get("entry") or {}
    raw_name = entry.get("name", "")
    cls = _strip_class_name(raw_name)
    cuda_us = float(entry.get("cuda_time_us", 0.0) or 0.0)

    matched = _match_against_slice(cls, ancestors, slice_)
    if matched is not None and cuda_us > 0:
        canonical, op_kind = matched
        out.append(TimingSample(
            layer=canonical,
            op_kind=op_kind,
            microseconds=cuda_us,
        ))

    new_ancestors = ancestors + [cls] if cls else ancestors
    for child in node.get("children", []) or []:
        _dfs(child, new_ancestors, slice_, out)


def _match_against_slice(
    node_class: str,
    ancestors: list[str],
    slice_: dict[str, dict[str, Any]],
) -> tuple[str, str] | None:
    """slice 里找命中 (node_class, ancestor 链) 的 entry, 返 (canonical, op_kind).

    歧义消解: within 在 ancestor 链中最深者胜, 无 within 视深度 -1 (排末).
    """
    candidates: list[tuple[int, str, str]] = []
    for canonical, fields in slice_.items():
        if fields.get("vllm") != node_class:
            continue
        within = fields.get("within")
        if within is None:
            candidates.append((-1, canonical, fields.get("op_kind", canonical)))
            continue
        try:
            depth = ancestors.index(within)
        except ValueError:
            continue
        candidates.append((depth, canonical, fields.get("op_kind", canonical)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, canonical, op_kind = candidates[0]
    return canonical, op_kind
