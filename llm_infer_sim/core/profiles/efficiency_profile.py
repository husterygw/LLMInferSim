"""EfficiencyProfile — per-op efficiency lookup interface (no longer YAML-backed).

历史:
  - 早期 (阶段 X.1) 这里支持 YAML I/O, 从 `configs/efficiency/<hw>.yaml` 读取
    端到端 gap 反推得到的 efficiency 系数。
  - 2026-05-18 实测发现这条数据生产线违反校准方法论 §2.1 铁律 1
    ("独立测量优先, 不端到端反推"), eager / graph 模式互不兼容, 已 retire。
  - 接口本身保留(EfficiencyProfile dataclass + lookup), 等待未来 MeasuredOperatorDB
    填入真测数据。当前默认 placeholder() = 全 1.0 = pure roofline 上界。

调用方:
  - `core/cost_model/roofline.py` 在 `analyze(op)` 后用 lookup_entry() 查 efficiency
    并 multiply 到 t_compute/t_memory/inference_time。
  - 当前所有 entry dict 为空, lookup_entry() 总返 None, roofline 走纯公式路径。

未来扩展(Operator DB 集成):
  - 当 MeasuredOperatorDB 落地时, 由它 populate self.entries (而非 YAML);
  - 或者 cost_model 路径直接 query MeasuredOperatorDB 拿 absolute latency,
    跳过 efficiency ratio 中间层。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EfficiencyEntry:
    """单条 efficiency 条目: (op_kind, dtype, shape_key) → efficiency。

    efficiency 定义: roofline_predicted_us / real_measured_us。
    > 1 表示 cost model 公式低估 (kernel 跑得比 roofline 还快);
    < 1 表示 kernel 没跑满 roofline 上界 (常见, 0.3-0.95 之间)。
    """
    op_kind: str          # 例 "dense_gemm" / "attn_prefill" / "rmsnorm" / "rope"
    dtype: str            # 例 "bf16" / "fp16" / "fp8" / "fp4"
    shape_key: str        # 例 "tokens<=128" / "ctx_4k" / "*" (通配)
    efficiency: float
    confidence: float = 1.0
    n_samples: int = 1
    source: str = ""      # provenance: e.g. "measured_db_v1/RTX_4090/2026-Q3"

    def key(self) -> tuple[str, str, str]:
        return (self.op_kind, self.dtype, self.shape_key)


@dataclass
class EfficiencyProfile:
    """Per-op efficiency lookup table。当前默认 placeholder (全 1.0)."""

    # 默认 fallback (没 entry 时用)
    default_compute: float = 1.0
    default_mem: float = 1.0
    default_comm: float = 1.0

    # bytes per element (跟随系统量化设置, 默认 fp16=2.0)
    # extract_profile_bundle 用这些字段决定 DeployConfig.w_byte/a_byte/kv_byte。
    w_byte: float = 2.0
    a_byte: float = 2.0
    kv_byte: float = 2.0

    # 查表
    entries: dict[tuple[str, str, str], EfficiencyEntry] = field(default_factory=dict)

    # metadata (来源 / 数据集版本等)
    hardware: str = ""
    captured_at: str = ""
    source_version: str = ""

    @property
    def w_bit(self) -> int:
        return int(self.w_byte * 8)

    @property
    def a_bit(self) -> int:
        return int(self.a_byte * 8)

    @property
    def kv_bit(self) -> int:
        return int(self.kv_byte * 8)

    @classmethod
    def placeholder(cls) -> "EfficiencyProfile":
        """默认: 全 1.0 + fp16, 无 entries → cost model 走 pure roofline。"""
        return cls()

    def apply_to(self, hw) -> None:
        """把 default_* efficiency 应用到 HardwareConfig (全局 fallback)."""
        hw.compute_efficiency = self.default_compute
        hw.mem_efficiency = self.default_mem
        hw.comm_efficiency = self.default_comm

    def add_entry(self, entry: EfficiencyEntry) -> None:
        self.entries[entry.key()] = entry

    def lookup_entry(
        self,
        op_kind: str,
        dtype: str,
        shape_key: str,
    ) -> EfficiencyEntry | None:
        """查 (op_kind, dtype, shape_key) 的 EfficiencyEntry, miss 返 None.

        4 级 wildcard 匹配顺序: 精确 > shape-* > dtype-* > op-only.
        miss 返 None — caller 用这个判断是否要 fallback 到 hw scalar default.
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
        """查 efficiency, miss fallback 到 category default。"""
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
