"""MemoryOperatorStore — in-memory dict, 单测/小规模调试用."""
from __future__ import annotations

from typing import Iterator

from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_schema.signature import OperatorSignature


class MemoryOperatorStore:
    """dict-based store. stable_hash 当 dict key."""

    def __init__(self):
        self._records: dict[str, OperatorRecord] = {}

    def add(self, record: OperatorRecord) -> None:
        """后写入覆盖前者 (高 confidence 由 importer 决定写入顺序; 这里不做合并)."""
        self._records[record.signature.stable_hash()] = record

    def lookup(self, signature: OperatorSignature) -> OperatorRecord | None:
        return self._records.get(signature.stable_hash())

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[OperatorRecord]:
        return iter(self._records.values())
