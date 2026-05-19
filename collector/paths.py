"""数据目录路径布局.

```
<data_root>/                                  e.g. collector/data/
  operator_db/<HW>/<framework>-<version>/     e.g. RTX_4090/vllm-0.19.1/
    <op>.jsonl                                main 测量结果
    errors/<op>.jsonl                         失败 case
    checkpoints/<op>.json                     per-op resume state
    progress.jsonl                            跨 op 总进度 (append-only)
    manifest.yaml                             采集环境快照
```
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from collector.schemas import Framework, OpKind


@dataclass(frozen=True)
class DataPaths:
    """所有 collector 输出路径的统一计算器.

    通过 `from_args(data_root, hw, framework, version)` 构造, 一份 path 对象对应
    一个 (hw, framework, version) 组合.
    """
    base: Path     # e.g. /.../collector/data/operator_db/RTX_4090/vllm-0.19.1

    @classmethod
    def from_args(
        cls,
        data_root: str | Path,
        hardware: str,
        framework: Framework,
        framework_version: str,
    ) -> "DataPaths":
        root = (
            Path(data_root)
            / "operator_db"
            / hardware
            / f"{framework.value}-{framework_version}"
        )
        return cls(base=root)

    # ---- file paths ----

    def op_jsonl(self, op: OpKind) -> Path:
        return self.base / f"{op.value}.jsonl"

    def errors_jsonl(self, op: OpKind) -> Path:
        return self.base / "errors" / f"{op.value}.jsonl"

    def checkpoint_json(self, op: OpKind) -> Path:
        return self.base / "checkpoints" / f"{op.value}.json"

    @property
    def progress_jsonl(self) -> Path:
        return self.base / "progress.jsonl"

    @property
    def manifest_yaml(self) -> Path:
        return self.base / "manifest.yaml"

    # ---- ensure directories exist ----

    def ensure_dirs(self) -> None:
        """创建所有需要的子目录 (idempotent)."""
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / "errors").mkdir(exist_ok=True)
        (self.base / "checkpoints").mkdir(exist_ok=True)
