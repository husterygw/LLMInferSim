"""EfficiencyProfile — 硬件效率系数表 (详设 §9.4.2 Plan B)。

阶段 1-9 : `placeholder()` 全 1.0, cost = 纯 roofline 上界。
阶段 X.1 起 : 升级为 `(op_kind, dtype, shape_key)` lookup table。

向后兼容: 旧路径 (placeholder) 没变, 新查表路径 miss 时 fallback 到 `default_compute`。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EfficiencyEntry:
    """单条校准条目: (op_kind, dtype, shape_key) → efficiency。

    efficiency 定义: roofline_predicted_us / real_measured_us, 范围理论 (0, 1].
    > 1 表示我们 cost model 公式低估了 (例 FA kernel 在某 shape 上跑得比 roofline 还快, 内部用 tensor core 多次 reuse);
    < 1 表示真 kernel 没跑满 roofline 上界 (常见, 大部分 op 在 0.3-0.95 之间).
    """
    op_kind: str          # 例 "dense_gemm" / "attn_prefill" / "rmsnorm" / "rope"
    dtype: str            # 例 "bf16" / "fp16" / "fp8" / "fp4"
    shape_key: str        # 例 "tokens<=128" / "ctx_4k" / "*" (通配)
    efficiency: float
    confidence: float = 1.0
    n_samples: int = 1
    source: str = ""      # provenance: e.g. "rtx_4090/Qwen3-4B/bf16/2026-05-15"

    def key(self) -> tuple[str, str, str]:
        return (self.op_kind, self.dtype, self.shape_key)


@dataclass
class EfficiencyProfile:
    """阶段 X.1 lookup-table version. 阶段 1-9 placeholder() 行为不变。"""

    # 默认 fallback (没 entry 时用; 旧 compute/mem/comm_efficiency 语义)
    default_compute: float = 1.0
    default_mem: float = 1.0
    default_comm: float = 1.0

    # bytes per element (跟随系统量化设置, 阶段 1 默认 fp16)
    # 这些字段仅作 backward-compat 桥: extract_profile_bundle 用它们决定
    # DeployConfig.w_byte / a_byte / kv_byte。Calibration 不动它们。
    w_byte: float = 2.0
    a_byte: float = 2.0
    kv_byte: float = 2.0

    # 查表 (阶段 X.1+)
    entries: dict[tuple[str, str, str], EfficiencyEntry] = field(default_factory=dict)

    # metadata (从 YAML 加载时填)
    hardware: str = ""
    captured_at: str = ""
    vllm_version: str = ""

    @property
    def w_bit(self) -> int:
        return int(self.w_byte * 8)

    @property
    def a_bit(self) -> int:
        return int(self.a_byte * 8)

    @property
    def kv_bit(self) -> int:
        return int(self.kv_byte * 8)

    # ---- compat with stage 1-9 callers ----

    @classmethod
    def placeholder(cls) -> "EfficiencyProfile":
        """阶段 1 默认: 全 1.0 + fp16, 无 entries."""
        return cls()

    def apply_to(self, hw) -> None:
        """把 default_* efficiency 应用到 HardwareConfig (阶段 1-9 旧路径)。

        阶段 X.1+ 起, per-op efficiency 走 lookup() 路径在 RooflineAnalyzer 里
        独立应用; 这里只设全局 fallback。
        """
        hw.compute_efficiency = self.default_compute
        hw.mem_efficiency = self.default_mem
        hw.comm_efficiency = self.default_comm

    # ---- lookup (阶段 X.1+) ----

    def add_entry(self, entry: EfficiencyEntry) -> None:
        self.entries[entry.key()] = entry

    def lookup_entry(
        self,
        op_kind: str,
        dtype: str,
        shape_key: str,
    ) -> EfficiencyEntry | None:
        """查 (op_kind, dtype, shape_key) 的 EfficiencyEntry, miss 返 None.

        4 级 wildcard 匹配顺序跟 lookup() 一致 (精确 > shape-* > dtype-* > op-only).
        miss 时返 None (不像 lookup() 返默认值) — caller 用这个判断是否要 fallback
        到 hw scalar default, 防止双重应用 efficiency.
        """
        for key in (
            (op_kind, dtype, shape_key),
            (op_kind, dtype, "*"),
            (op_kind, "*", shape_key),
            (op_kind, "*", "*"),
        ):
            entry = self.entries.get(key)
            if entry is not None:
                return entry
        return None

    def lookup(
        self,
        op_kind: str,
        dtype: str,
        shape_key: str,
        category: str = "compute",
    ) -> float:
        """查 (op_kind, dtype, shape_key) 的 efficiency.

        匹配顺序 (前者优先):
          1. (op_kind, dtype, shape_key)          精确匹配
          2. (op_kind, dtype, "*")                shape 通配
          3. (op_kind, "*", shape_key)            dtype 通配
          4. (op_kind, "*", "*")                  op_kind only
          5. category fallback: default_compute / default_mem / default_comm

        Args:
            category: "compute" | "mem" | "comm", 决定 fallback 用哪个 default。
        """
        for key in (
            (op_kind, dtype, shape_key),
            (op_kind, dtype, "*"),
            (op_kind, "*", shape_key),
            (op_kind, "*", "*"),
        ):
            entry = self.entries.get(key)
            if entry is not None:
                return entry.efficiency
        if category == "mem":
            return self.default_mem
        if category == "comm":
            return self.default_comm
        return self.default_compute

    # ---- YAML I/O ----

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EfficiencyProfile":
        """读 configs/efficiency/<hw>.yaml.

        Schema:
          hardware: RTX_4090
          captured_at: 2026-05-15
          vllm_version: 0.20.1
          default_compute: 0.7
          default_mem: 0.85
          default_comm: 1.0
          w_byte: 2.0
          a_byte: 2.0
          kv_byte: 2.0
          entries:
            - {op_kind: dense_gemm, dtype: bf16, shape_key: "tokens<=128", efficiency: 0.62, ...}
            - ...

        缺字段时用 placeholder default. 路径不存在时 raise FileNotFoundError.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"EfficiencyProfile YAML not found: {path}")
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "EfficiencyProfile.from_yaml requires PyYAML. pip install pyyaml"
            ) from e
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        prof = cls(
            default_compute=float(data.get("default_compute", 1.0)),
            default_mem=float(data.get("default_mem", 1.0)),
            default_comm=float(data.get("default_comm", 1.0)),
            w_byte=float(data.get("w_byte", 2.0)),
            a_byte=float(data.get("a_byte", 2.0)),
            kv_byte=float(data.get("kv_byte", 2.0)),
            hardware=str(data.get("hardware", "")),
            captured_at=str(data.get("captured_at", "")),
            vllm_version=str(data.get("vllm_version", "")),
        )
        for raw in data.get("entries", []) or []:
            try:
                entry = EfficiencyEntry(
                    op_kind=str(raw["op_kind"]),
                    dtype=str(raw["dtype"]),
                    shape_key=str(raw["shape_key"]),
                    efficiency=float(raw["efficiency"]),
                    confidence=float(raw.get("confidence", 1.0)),
                    n_samples=int(raw.get("n_samples", 1)),
                    source=str(raw.get("source", "")),
                )
            except (KeyError, TypeError, ValueError) as e:
                warnings.warn(f"Skipping invalid EfficiencyEntry {raw!r}: {e}")
                continue
            prof.add_entry(entry)
        return prof

    def to_yaml(self, path: str | Path) -> None:
        """写 configs/efficiency/<hw>.yaml."""
        path = Path(path)
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "EfficiencyProfile.to_yaml requires PyYAML. pip install pyyaml"
            ) from e
        # entries 按 key 排序保证 YAML 稳定 (diff-friendly)
        sorted_entries = sorted(self.entries.values(), key=lambda e: e.key())
        data = {
            "hardware": self.hardware,
            "captured_at": self.captured_at,
            "vllm_version": self.vllm_version,
            "default_compute": self.default_compute,
            "default_mem": self.default_mem,
            "default_comm": self.default_comm,
            "w_byte": self.w_byte,
            "a_byte": self.a_byte,
            "kv_byte": self.kv_byte,
            "entries": [
                {
                    "op_kind": e.op_kind,
                    "dtype": e.dtype,
                    "shape_key": e.shape_key,
                    "efficiency": e.efficiency,
                    "confidence": e.confidence,
                    "n_samples": e.n_samples,
                    "source": e.source,
                }
                for e in sorted_entries
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
