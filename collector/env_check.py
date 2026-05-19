"""启动前 preflight — 采集环境快照 + sanity 检查.

输出: manifest.yaml (跟测量数据放一起, 审计必备).

设计:
  - 全部 detection 用 try/except 兜底, 缺一个 tool 不致命
  - 不强制要 GPU (CPU-only env 也能 import 不报错)
  - lock_freq / 独占 GPU 检查只 warn, 不阻断 (开发机不一定能锁)
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EnvSnapshot:
    """采集时机器环境快照. 写到 manifest.yaml."""
    # 软件版本
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    nccl_version: str = ""
    vllm_version: str = ""
    sglang_version: str = ""

    # 硬件
    gpu_name: str = ""           # 例: "NVIDIA GeForce RTX 4090"
    gpu_count: int = 0
    driver_version: str = ""
    gpu_lock_freq_mhz: int | None = None    # None = 未锁
    gpu_exclusive_compute_mode: str = ""    # "Default" / "Exclusive_Process" / ...

    # 元信息
    collector_version: str = ""
    captured_at: str = ""
    notes: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_yaml_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 各种 detect 函数 (全 try/except 兜底)
# ---------------------------------------------------------------------------

def _detect_python() -> str:
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _detect_pkg_version(pkg: str) -> str:
    try:
        mod = __import__(pkg)
        v = getattr(mod, "__version__", "")
        return str(v) if v else ""    # 强转 (torch 的 TorchVersion 等子类 → 纯 str)
    except Exception:
        return ""


def _detect_cuda_version() -> str:
    try:
        import torch
        v = torch.version.cuda
        return str(v) if v else ""
    except Exception:
        return ""


def _detect_nccl_version() -> str:
    try:
        import torch
        v = torch.cuda.nccl.version()  # (major, minor, patch)
        return ".".join(str(x) for x in v)
    except Exception:
        return ""


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip().splitlines()
        return out[0] if out else ""
    except Exception:
        return ""


def _detect_gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _detect_driver_version() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip().splitlines()
        return out[0] if out else ""
    except Exception:
        return ""


def _detect_gpu_freq_lock() -> int | None:
    """读 graphics clock lock. 未锁返 None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.applications.graphics",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip().splitlines()
        if not out:
            return None
        val = out[0].strip()
        # nvidia-smi 在没锁时返 "[Not Supported]" 或 0
        if not val or "Not" in val or "N/A" in val:
            return None
        return int(val)
    except Exception:
        return None


def _detect_compute_mode() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_mode", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip().splitlines()
        return out[0] if out else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 主入口 — preflight + EnvSnapshot
# ---------------------------------------------------------------------------

def collect_env(
    *,
    collector_version: str = "",
    notes: str = "",
    warn_unlocked_gpu: bool = True,
) -> EnvSnapshot:
    """采集当前环境快照. 不抛错, 缺字段返空."""
    snap = EnvSnapshot(
        python_version=_detect_python(),
        torch_version=_detect_pkg_version("torch"),
        cuda_version=_detect_cuda_version(),
        nccl_version=_detect_nccl_version(),
        vllm_version=_detect_pkg_version("vllm"),
        sglang_version=_detect_pkg_version("sglang"),
        gpu_name=_detect_gpu_name(),
        gpu_count=_detect_gpu_count(),
        driver_version=_detect_driver_version(),
        gpu_lock_freq_mhz=_detect_gpu_freq_lock(),
        gpu_exclusive_compute_mode=_detect_compute_mode(),
        collector_version=collector_version,
        captured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        notes=notes,
    )

    # 一些非致命 warning
    if warn_unlocked_gpu and snap.gpu_count > 0 and snap.gpu_lock_freq_mhz is None:
        snap.warnings.append(
            "GPU 频率未锁 — 数据可能有 ±5% 抖动. 生产采集请 `nvidia-smi -lgc <freq>`."
        )
    if snap.gpu_count > 0 and snap.gpu_exclusive_compute_mode == "Default":
        snap.warnings.append(
            "GPU 处于 Default compute mode — 其他进程可争用. 生产采集请 Exclusive_Process."
        )

    return snap


def write_manifest(snap: EnvSnapshot, path: Path) -> None:
    """写 manifest.yaml. 失败 raise (调用方决定)."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("env_check.write_manifest requires PyYAML") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(snap.to_yaml_dict(), sort_keys=False, allow_unicode=True)
    )


def auto_hardware_id(snap: EnvSnapshot) -> str:
    """从 GPU name 推 short id. 例: 'NVIDIA GeForce RTX 4090' → 'RTX_4090'."""
    name = snap.gpu_name.lower()
    if "4090" in name:
        return "RTX_4090"
    if "h100" in name:
        return "H100"
    if "h200" in name:
        return "H200"
    if "h800" in name:
        return "H800"
    if "a100" in name:
        return "A100"
    if "a6000" in name:
        return "A6000"
    if "b100" in name:
        return "B100"
    if "b200" in name:
        return "B200"
    # fallback: spaces to underscores
    return snap.gpu_name.replace(" ", "_") or "unknown"
