"""E2E smoke wrappers around examples/ — 薄 "documented command check"。

每个 example 脚本约定: exit code 0 + stdout 含 'PASSED' 即成功。这里不复制断言逻辑,
只带上 CLAUDE.md 规定的运行环境, subprocess 跑脚本, 校验退出码和成功标记。

默认全部 skip(见根 conftest.py 的自动 skip):
  - 所有用例标 `e2e`, 未 opt-in(RUN_E2E=1)时 skip。
  - 起真实 vLLM LLM 的用例额外标 `gpu`, 无 CUDA 时 skip。
  - 依赖本地模型的用例, 模型路径不存在时 skip(可用 VLLM_INFER_SIM_MODEL 覆盖)。

跑法:
    RUN_E2E=1 pytest tests/e2e -m e2e -q                # 全部(platform 探针不需 GPU)
    RUN_E2E=1 pytest tests/e2e -m "e2e and gpu" -q      # 仅起真实 vLLM 的
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"

# CLAUDE.md: 跑模拟器必加这组 env, 缺一个就挂。
SIM_ENV = {
    "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    "VLLM_VIRTUAL_BACKEND": "1",
    "VLLM_USE_V1": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

DEFAULT_QWEN = "/data1/home/ygw268/models/Qwen3-4B-Instruct-2507"


def _needs_local_model(default: str):
    path = os.environ.get("VLLM_INFER_SIM_MODEL", default)
    return pytest.mark.skipif(
        not Path(path).exists(),
        reason=f"local model not found: {path} (set VLLM_INFER_SIM_MODEL)",
    )


def _run_example(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script)],
        env={**os.environ, **SIM_ENV},
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize(
    "script",
    [
        pytest.param("vllm_virtual/run_platform_selected.py", marks=pytest.mark.e2e,
                     id="platform_selected"),
        pytest.param("vllm_virtual/run_opt125m.py",
                     marks=[pytest.mark.e2e, pytest.mark.gpu], id="opt125m"),
        pytest.param("vllm_virtual/run_qwen3_4b.py",
                     marks=[pytest.mark.e2e, pytest.mark.gpu,
                            pytest.mark.skipif(
                                not Path(DEFAULT_QWEN).exists(),
                                reason=f"local model not found: {DEFAULT_QWEN}")],
                     id="qwen3_4b"),
        pytest.param("vllm_virtual/run_prefix_caching.py",
                     marks=[pytest.mark.e2e, pytest.mark.gpu,
                            _needs_local_model(DEFAULT_QWEN)], id="prefix_caching"),
        pytest.param("pd_disagg/run_pd_disagg_loopback.py",
                     marks=[pytest.mark.e2e, pytest.mark.gpu,
                            _needs_local_model(DEFAULT_QWEN)], id="pd_loopback"),
    ],
)
def test_example_smoke(script):
    proc = _run_example(script)
    tail = f"\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
    assert proc.returncode == 0, f"{script} exited {proc.returncode}{tail}"
    assert "PASSED" in proc.stdout, f"{script}: missing PASSED marker{tail}"
