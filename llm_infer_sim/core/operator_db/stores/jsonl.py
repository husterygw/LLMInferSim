"""JsonlOperatorStore — 从 collector/data/operator_db JSONL 加载.

路径约定 (collector convention):
    <root>/<hardware>/<framework>-<framework_version>/<op_kind>.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from llm_infer_sim.core.operator_db.importers.collector_v2 import import_record
from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_db.stores.memory import MemoryOperatorStore


class JsonlOperatorStore(MemoryOperatorStore):
    """MemoryOperatorStore + JSONL loader. 兼容同样的 lookup/add 接口."""

    @classmethod
    def from_jsonl(cls, path: Path | str, *, hardware: str) -> "JsonlOperatorStore":
        store = cls()
        store.load_jsonl(path, hardware=hardware)
        return store

    def load_jsonl(self, path: Path | str, *, hardware: str) -> int:
        """从一个 JSONL file 加载. 返成功导入条数."""
        p = Path(path)
        count = 0
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                try:
                    rec = import_record(row, hardware=hardware)
                except Exception as e:
                    # Skip malformed rows but log via raise during dev would be cleaner;
                    # stage 3 keep tolerant so partial JSONL doesn't break load.
                    continue
                self.add(rec)
                count += 1
        return count

    def load_partition(
        self,
        root: Path | str,
        *,
        hardware: str,
        framework: str,
        framework_version: str,
        op_kinds: tuple[str, ...] = ("gemm", "attention", "moe", "collective"),
    ) -> dict[str, int]:
        """加载某个 (hardware, framework, version) 分区下指定 op_kinds 的 JSONL.

        路径布局: <root>/<hardware>/<framework>-<framework_version>/<op_kind>.jsonl
        返 {op_kind: count} 实际加载条数.
        """
        partition_dir = Path(root) / hardware / f"{framework}-{framework_version}"
        counts: dict[str, int] = {}
        for op_kind in op_kinds:
            jsonl_path = partition_dir / f"{op_kind}.jsonl"
            if jsonl_path.exists():
                counts[op_kind] = self.load_jsonl(jsonl_path, hardware=hardware)
            else:
                counts[op_kind] = 0
        return counts
