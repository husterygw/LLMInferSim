"""V3 §8.2 OperatorStore — DB 抽象接口."""
from __future__ import annotations

from typing import Protocol

from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_schema.signature import OperatorSignature


class OperatorStore(Protocol):
    """OperatorRecord 的 lookup 抽象. Stage 3 实现 MemoryOperatorStore + JsonlOperatorStore."""

    def add(self, record: OperatorRecord) -> None: ...

    def lookup(self, signature: OperatorSignature) -> OperatorRecord | None: ...

    def __len__(self) -> int: ...
