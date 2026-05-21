"""V3 §5 + IMPL_PLAN §2.1 OperatorSignature — collector / runtime / OperatorDB 共用 key.

contract (硬规则):
    collector Case.params
    runtime Operator.shape/parallel/runtime
    OperatorDB OperatorRecord.signature

    必须能 canonicalize 成同一个 OperatorSignature.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OperatorSignature:
    """Hashable + JSON-serializable op identity.

    shape / parallel / runtime 是 tuple-of-(str, value), 保证 frozen 可 hash;
    tuple 顺序由 canonical (alphabetical) 排序锁定, 保证同语义 dict 入参得到同 signature.
    """
    op_kind: str
    op_subtype: str
    dtype: str
    shape: tuple[tuple[str, Any], ...]
    parallel: tuple[tuple[str, Any], ...]
    runtime: tuple[tuple[str, Any], ...]

    def stable_hash(self) -> str:
        """跨进程稳定的 hex digest, OperatorDB lookup key 用."""
        return hashlib.sha256(
            json.dumps(self.to_json_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        """JSON serializable dict (供 reporter / DB store / debug print)."""
        return {
            "op_kind": self.op_kind,
            "op_subtype": self.op_subtype,
            "dtype": self.dtype,
            "shape": [list(item) for item in self.shape],
            "parallel": [list(item) for item in self.parallel],
            "runtime": [list(item) for item in self.runtime],
        }
