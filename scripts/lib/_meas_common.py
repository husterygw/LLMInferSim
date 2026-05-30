"""Shared helpers for scripts/measure_*.py — version-aware JSONL output.

每个 measure_*.py 都应该用 `make_record()` 给 JSON 加 version metadata,
方便后续 audit / 复现 (詳 docs/CALIBRATION_METHODOLOGY.md §10).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _safe_pkg(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return "?"


def _nvidia_driver() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip().splitlines()
        return out[0] if out else "?"
    except Exception:
        return "?"


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip().splitlines()
        return out[0] if out else "?"
    except Exception:
        return "?"


def env_metadata() -> dict:
    """采集机器 + 软件版本, 写到每条 measurement record."""
    return {
        "gpu_name": _gpu_name(),
        "driver_version": _nvidia_driver(),
        "torch_version": _safe_pkg("torch"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def make_record(script_name: str, **fields) -> dict:
    """组装一条 JSONL record: script meta + measurement fields."""
    return {
        "script": script_name,
        **env_metadata(),
        **fields,
    }


def write_jsonl(path: str | Path, records: list[dict]) -> Path:
    """写 JSONL, 自动建父目录."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def default_out_path(script_name: str, hw_id: str | None = None) -> Path:
    """默认输出位置: configs/calibration/raw/<HW>/<YYYY-MM-DD>/<script>.jsonl."""
    if hw_id is None:
        # 从 GPU name 提取一个 short id, e.g. "RTX_4090"
        gpu = _gpu_name()
        if "4090" in gpu:
            hw_id = "RTX_4090"
        elif "H100" in gpu:
            hw_id = "H100"
        elif "A100" in gpu:
            hw_id = "A100"
        else:
            hw_id = gpu.replace(" ", "_") or "unknown"
    date = time.strftime("%Y-%m-%d")
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs" / "calibration" / "raw" / hw_id / date / f"{script_name}.jsonl"


def print_records_table(records: list[dict], cols: list[str]) -> None:
    """简单表格打印到 stdout."""
    if not records:
        return
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in records)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in records:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
