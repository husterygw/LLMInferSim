"""全局 (op, framework) → CollectorEntry 注册表.

启动时由 entries() 集中注册, scheduler / cli 从 REGISTRY 查询.

设计原则:
  - 注册表是数据驱动 (CollectorEntry 数据描述), 不是反射
  - 一个 (op, framework) 组合一个 entry, 重复注册 raise (防 silent override)
  - 不在 import 时副作用注册, 由 cli.py 显式调 `bootstrap()` 注册
"""
from __future__ import annotations

from collector.schemas import CollectorEntry, Framework, OpKind


class CollectorRegistry:
    """In-memory 注册表. (op, framework) → CollectorEntry."""

    def __init__(self):
        self._entries: dict[tuple[OpKind, Framework], CollectorEntry] = {}

    def register(self, entry: CollectorEntry) -> None:
        """注册一个 entry. 同 key 重复 raise."""
        key = (entry.op, entry.framework)
        if key in self._entries:
            raise ValueError(
                f"Duplicate registry entry for ({entry.op.value}, "
                f"{entry.framework.value}): existing={self._entries[key]!r}"
            )
        self._entries[key] = entry

    def get(self, op: OpKind, framework: Framework) -> CollectorEntry | None:
        """查 entry. 未注册返 None."""
        return self._entries.get((op, framework))

    def require(self, op: OpKind, framework: Framework) -> CollectorEntry:
        """查 entry, 未注册 raise."""
        entry = self.get(op, framework)
        if entry is None:
            raise KeyError(
                f"No registry entry for op={op.value}, framework={framework.value}. "
                f"Available: {[(o.value, f.value) for o, f in self._entries.keys()]}"
            )
        return entry

    def list_ops(self, framework: Framework | None = None) -> list[OpKind]:
        """列出已注册的 op. framework 给定则只列那个框架的."""
        if framework is None:
            return sorted({op for (op, _) in self._entries.keys()}, key=lambda x: x.value)
        return sorted(
            [op for (op, fw) in self._entries.keys() if fw == framework],
            key=lambda x: x.value,
        )

    def list_frameworks(self) -> list[Framework]:
        """列出已注册的 framework."""
        return sorted({fw for (_, fw) in self._entries.keys()}, key=lambda x: x.value)

    def all_entries(self) -> list[CollectorEntry]:
        """全部 entries (按 (op, framework) 排序)."""
        return [
            self._entries[k]
            for k in sorted(self._entries.keys(), key=lambda x: (x[0].value, x[1].value))
        ]

    def clear(self) -> None:
        """测试用."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple[OpKind, Framework]) -> bool:
        return key in self._entries


# 全局 singleton — 由 cli.bootstrap() 显式 register
REGISTRY = CollectorRegistry()
