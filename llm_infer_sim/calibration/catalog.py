"""模型 catalog — vLLM nn.Module 类名 + ancestor → canonical op (详设 §9.4.2 B.1).

YAML schema (例 models/qwen3.yaml):

    model_type: qwen3
    description: ...
    entries:
      qkv_proj:
        vllm: QKVParallelLinear
        op_kind: dense_gemm
      layernorm:
        vllm: RMSNorm
        within: Qwen3DecoderLayer        # 必须在 Qwen3DecoderLayer 内部
        op_kind: rmsnorm
      qk_norm:
        vllm: RMSNorm
        within: Qwen3Attention           # 必须在 Qwen3Attention 内部
        op_kind: rmsnorm
      ...

匹配规则: catalog entry 命中 node 当且仅当
  (1) node 类名 == entry.vllm
  (2) entry.within is None 或 某 ancestor 类名 == entry.within

歧义消解: 多个 entry 都命中同一 node 时, within 最深的 (在 ancestor 链中位置最近 node)
  胜出。无 within 的 entry 排在最末。例 Qwen3 RMSNorm 在 DecoderLayer (input/post-attn
  layernorm) 和 Attention (qk_norm) 都有, 内部 attention 节点应优先匹 qk_norm 而不是
  外层 layernorm。这条规则跟 LLMServingSim profiler/core/hooks/timings.py 对齐。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CatalogEntry:
    """一条 canonical → vLLM 映射条目.

    category: 决定该 canonical 进哪个 shot kind 的 CSV.
        "dense":        token-count-driven (qkv_proj / norm / mlp / embedding / ...).
        "attention":    attention kernel (4D shape: prefill_chunk / kv_lens / n_decode).
        "per_sequence": #seqs-driven (lm_head / sampler).
    """
    canonical: str
    vllm_class: str
    within: str | None       # ancestor class name; None = 任意
    op_kind: str             # efficiency lookup 用的 op_kind
    category: str = "dense"  # shot kind 分类, YAML 默认 dense

    def to_slice_dict(self) -> dict[str, Any]:
        """跨进程序列化形式 (传 worker_extension.fire)."""
        return {
            "canonical": self.canonical,
            "vllm": self.vllm_class,
            "within": self.within,
            "op_kind": self.op_kind,
            "category": self.category,
        }


class Catalog:
    """加载 model_type catalog YAML, 提供 slice + match 接口."""

    def __init__(self, model_type: str, entries: list[CatalogEntry]):
        self.model_type = model_type
        self.entries: list[CatalogEntry] = list(entries)

    @classmethod
    def load(cls, model_type: str, models_dir: Path | None = None) -> "Catalog":
        """读 `<models_dir>/<model_type>.yaml`. models_dir 默认 calibration/models/.

        缺失 YAML 时 raise FileNotFoundError. 用户应在 calibration/models/ 加新的.
        """
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("Catalog YAML 需要 PyYAML. pip install pyyaml") from e

        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        path = models_dir / f"{model_type}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Catalog YAML 未找到: {path}. "
                f"新模型族 model_type='{model_type}' 需要在 calibration/models/ 添加 YAML."
            )

        data = yaml.safe_load(path.read_text()) or {}
        raw_entries: dict[str, dict[str, Any]] = data.get("entries", {}) or {}
        parsed: list[CatalogEntry] = []
        for canonical, fields in raw_entries.items():
            try:
                # category 默认 rule: attention 类名走 "attention", lm_head/sampler 走
                # "per_sequence", 其他走 "dense". YAML 显式给 category 覆盖默认.
                op_kind = str(fields.get("op_kind", canonical))
                default_cat = _default_category(canonical, op_kind)
                parsed.append(CatalogEntry(
                    canonical=canonical,
                    vllm_class=str(fields["vllm"]),
                    within=fields.get("within") or None,
                    op_kind=op_kind,
                    category=str(fields.get("category", default_cat)),
                ))
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"catalog entry {canonical!r} in {path}: {e}"
                ) from e
        return cls(model_type=model_type, entries=parsed)

    # ---- 匹配 ----

    def match(self, node_class: str, ancestors: list[str]) -> CatalogEntry | None:
        """node 类名 + ancestor 链 → canonical entry. miss 时返 None.

        ancestors[0] 是外层 (root), ancestors[-1] 是最近 parent.
        歧义消解: within 最深者胜 (即 within 在 ancestors 里位置最大者).
                  无 within 的 entry 视为"深度 -1" (排末).
        """
        candidates: list[tuple[int, CatalogEntry]] = []
        for e in self.entries:
            if e.vllm_class != node_class:
                continue
            if e.within is None:
                candidates.append((-1, e))
                continue
            try:
                depth = ancestors.index(e.within)
            except ValueError:
                continue                # within 不在 ancestor 链 → 不匹
            candidates.append((depth, e))
        if not candidates:
            return None
        # 深度最大者胜
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # ---- slice (传给 worker_extension) ----

    def slice_for_op_kinds(self, op_kinds: set[str] | None = None) -> dict[str, dict[str, Any]]:
        """返回 `{canonical: {"vllm": ..., "within": ..., "op_kind": ...}}` 字典.

        op_kinds 给定时只导出这些 op_kind 的 entry; None 时全部.
        """
        out: dict[str, dict[str, Any]] = {}
        for e in self.entries:
            if op_kinds is not None and e.op_kind not in op_kinds:
                continue
            out[e.canonical] = e.to_slice_dict()
        return out

    def slice_for_category(self, category: str) -> dict[str, dict[str, Any]]:
        """返回该 category 的 canonical 子集, 给 fire() 用."""
        return {
            e.canonical: e.to_slice_dict()
            for e in self.entries
            if e.category == category
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


# ---- 默认 category 推导规则 ----

# 显式映射的 op_kind → category. 其他全 dense.
_PER_SEQUENCE_CANONICALS = frozenset({"lm_head", "sampler"})


def _default_category(canonical: str, op_kind: str) -> str:
    """从 canonical / op_kind 推 category. YAML 显式给 category 时不调此函数."""
    if op_kind == "attn":
        return "attention"
    if canonical in _PER_SEQUENCE_CANONICALS:
        return "per_sequence"
    return "dense"
