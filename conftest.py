"""Root pytest config: 让裸 `pytest` 默认就干净 (test_plan.md §8/§10 Phase 0)。

按 marker 自动 skip 环境敏感测试, 这样不靠记长 `-m "not gpu and not e2e..."` 命令:
  - gpu       : 无 CUDA 时 skip
  - e2e       : 未 opt-in (RUN_E2E=1) 时 skip
  - realdata  : 未 opt-in (RUN_REALDATA=1) 时 skip
  - nightly   : 未 opt-in (RUN_NIGHTLY=1) 时 skip

slow 不在此 (与环境无关), 由 PR gate 命令显式 `-m "not slow"` 处理。
"""
from __future__ import annotations

import os

import pytest


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="requires CUDA/GPU (none available)")
    optin = {
        "e2e": ("RUN_E2E", pytest.mark.skip(reason="e2e opt-in: set RUN_E2E=1")),
        "realdata": (
            "RUN_REALDATA",
            pytest.mark.skip(reason="realdata opt-in: set RUN_REALDATA=1"),
        ),
        "nightly": (
            "RUN_NIGHTLY",
            pytest.mark.skip(reason="nightly opt-in: set RUN_NIGHTLY=1"),
        ),
    }
    cuda = _cuda_available()
    for item in items:
        if item.get_closest_marker("gpu") and not cuda:
            item.add_marker(skip_gpu)
        for marker_name, (env, skip_marker) in optin.items():
            if item.get_closest_marker(marker_name) and os.environ.get(env) != "1":
                item.add_marker(skip_marker)
