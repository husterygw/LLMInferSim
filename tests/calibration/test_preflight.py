"""preflight.py — calibration 前快速健康检查 (B.4 prep)."""
from __future__ import annotations

import sys
from unittest import mock

import pytest

from llm_infer_sim.calibration.preflight import main, preflight


def test_preflight_blocks_when_virtual_backend_set(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_VIRTUAL_BACKEND", "1")
    rc = preflight(model="x", model_type="qwen3", hardware="RTX_4090")
    assert rc == 1
    out = capsys.readouterr().out
    assert "VLLM_VIRTUAL_BACKEND=1" in out


def test_preflight_fails_no_gpu(monkeypatch, capsys):
    """torch.cuda 不可用 → rc=2."""
    monkeypatch.delenv("VLLM_VIRTUAL_BACKEND", raising=False)
    # mock torch.cuda.is_available → False
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rc = preflight(model="x", model_type="qwen3", hardware="RTX_4090")
    assert rc == 2
    out = capsys.readouterr().out
    assert "torch.cuda.is_available" in out


def test_main_parses_args(monkeypatch):
    """main() 跑通 argparse, 调 preflight()."""
    monkeypatch.delenv("VLLM_VIRTUAL_BACKEND", raising=False)
    # mock preflight 防真启动
    called = {}

    def fake_preflight(**kwargs):
        called.update(kwargs)
        return 0

    monkeypatch.setattr(
        "llm_infer_sim.calibration.preflight.preflight", fake_preflight,
    )
    rc = main([
        "--model", "Qwen/Qwen3-4B",
        "--model-type", "qwen3",
        "--hardware", "RTX_4090",
        "--test-tokens", "16",
    ])
    assert rc == 0
    assert called["model"] == "Qwen/Qwen3-4B"
    assert called["test_tokens"] == 16
