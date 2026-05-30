"""可选 YAML 薄覆盖层 (adapter 边界)。

定位:
  - 只表达对 "vLLM 推导值" 和 "现有 profile" 的**覆盖意图**, 不是第三套领域模型。
  - 优先级固定: vLLM 推导 < configs/config.yaml < 已存在 env。
  - 文件可选: 不存在 → 空 overlay → 当前行为完全不变。
  - 读磁盘路径属摄取/入口职责, 故放 vLLM adapter 层, core 保持框架无关。

env 用 presence detection (`os.environ.get(key)` is not None) 而非 `.get(key, default)`,
否则 env 永远返回默认值、YAML 永远生效不了。覆盖点见 profile_extractor / virtual_model_runner。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 默认路径: <repo>/configs/config.yaml。本文件在 llm_infer_sim/adapters/vllm/ 下,
# parents[3] = LLMInferSim repo root。
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.yaml"

_AUTO = "auto"


@dataclass
class HardwareOverlay:
    name: str | None = None
    topology_hint: str | None = None
    compute_efficiency: float | None = None
    mem_efficiency: float | None = None
    comm_efficiency: float | None = None


@dataclass
class CalibrationOverlay:
    enabled: bool | None = None


@dataclass
class QuantizationOverlay:
    # None 表示 "auto" (沿用 vLLM 推导); 只有 float 数值才覆盖 QuantizationProfile。
    w_byte: float | None = None
    a_byte: float | None = None
    kv_byte: float | None = None


@dataclass
class PDDisaggOverlay:
    role: str | None = None
    connector_name: str | None = None
    kv_parallel_size: int | None = None
    connector_bandwidth_gbps: float | None = None
    connector_latency_us: float | None = None


@dataclass
class RuntimeOverlay:
    time_mode: str | None = None
    dump_ops: int | None = None
    dump_requests: int | None = None


@dataclass
class SimOverlay:
    hardware: HardwareOverlay = field(default_factory=HardwareOverlay)
    calibration: CalibrationOverlay = field(default_factory=CalibrationOverlay)
    quantization: QuantizationOverlay = field(default_factory=QuantizationOverlay)
    pd_disagg: PDDisaggOverlay = field(default_factory=PDDisaggOverlay)
    runtime: RuntimeOverlay = field(default_factory=RuntimeOverlay)


# ---- scalar validators (fail-fast with clear context) ----
def _as_str(v, ctx):
    if not isinstance(v, str):
        raise ValueError(f"config.yaml: {ctx} 期望 string, 实际 {type(v).__name__}")
    return v


def _as_float(v, ctx):
    # 注意: bool 是 int 的子类, 显式排除, 不让 true/false 被当数值。
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"config.yaml: {ctx} 期望 number, 实际 {type(v).__name__}")
    return float(v)


def _as_int(v, ctx):
    # dump_ops/dump_requests: 允许 bool (false→0/true→1) 和 int (0/1/2)。
    if isinstance(v, bool):
        return int(v)
    if not isinstance(v, int):
        raise ValueError(f"config.yaml: {ctx} 期望 integer, 实际 {type(v).__name__}")
    return v


def _as_bool(v, ctx):
    if not isinstance(v, bool):
        raise ValueError(f"config.yaml: {ctx} 期望 bool, 实际 {type(v).__name__}")
    return v


def _as_float_or_auto(v, ctx):
    # auto 是 quantization byte 字段独有的哨兵: 沿用 vLLM 推导 (→ None, 不覆盖)。
    if v == _AUTO:
        return None
    return _as_float(v, ctx)


_SECTION_PARSERS = {
    "hardware": (HardwareOverlay, {
        "name": _as_str,
        "topology_hint": _as_str,
        "compute_efficiency": _as_float,
        "mem_efficiency": _as_float,
        "comm_efficiency": _as_float,
    }),
    "calibration": (CalibrationOverlay, {
        "enabled": _as_bool,
    }),
    "quantization": (QuantizationOverlay, {
        "w_byte": _as_float_or_auto,
        "a_byte": _as_float_or_auto,
        "kv_byte": _as_float_or_auto,
    }),
    "pd_disagg": (PDDisaggOverlay, {
        "role": _as_str,
        "connector_name": _as_str,
        "kv_parallel_size": _as_int,
        "connector_bandwidth_gbps": _as_float,
        "connector_latency_us": _as_float,
    }),
    "runtime": (RuntimeOverlay, {
        "time_mode": _as_str,
        "dump_ops": _as_int,
        "dump_requests": _as_int,
    }),
}


def _parse_overlay(raw) -> SimOverlay:
    if raw is None:
        return SimOverlay()
    if not isinstance(raw, dict):
        raise ValueError(
            f"config.yaml: 顶层必须是 mapping, 实际 {type(raw).__name__}"
        )
    unknown = set(raw) - set(_SECTION_PARSERS)
    if unknown:
        raise ValueError(
            f"config.yaml: 未知 section {sorted(unknown)}; "
            f"已知: {sorted(_SECTION_PARSERS)}"
        )
    sections = {}
    for name, (cls, parsers) in _SECTION_PARSERS.items():
        body = raw.get(name)
        if body is None:
            sections[name] = cls()
            continue
        if not isinstance(body, dict):
            raise ValueError(
                f"config.yaml: section {name!r} 必须是 mapping, "
                f"实际 {type(body).__name__}"
            )
        unknown_keys = set(body) - set(parsers)
        if unknown_keys:
            raise ValueError(
                f"config.yaml: section {name!r} 未知 key {sorted(unknown_keys)}; "
                f"已知: {sorted(parsers)}"
            )
        # value 为 None (YAML `key: null`) 视同未设 → 不覆盖。
        kwargs = {
            k: parsers[k](v, f"{name}.{k}")
            for k, v in body.items()
            if v is not None
        }
        sections[name] = cls(**kwargs)
    return SimOverlay(**sections)


# 进程内缓存: 配置变更需重启进程生效。key = 解析后的绝对路径。
_CACHE: dict[str, SimOverlay] = {}


def load_sim_overlay(path: Path | str | None = None) -> SimOverlay:
    """加载 overlay; 文件缺失返回空 overlay。进程内按路径缓存。"""
    p = Path(path) if path is not None else _DEFAULT_PATH
    key = str(p)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    if not p.exists():
        overlay = SimOverlay()
    else:
        with p.open("r", encoding="utf-8") as f:
            overlay = _parse_overlay(yaml.safe_load(f))
    _CACHE[key] = overlay
    return overlay


def _clear_cache() -> None:
    """测试用: 清进程内缓存。"""
    _CACHE.clear()
