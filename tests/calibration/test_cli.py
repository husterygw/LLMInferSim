"""__main__.py CLI (B.3)."""
from __future__ import annotations

import sys
from unittest import mock

import pytest

from llm_infer_sim.calibration.__main__ import _build_parser, main


def test_parser_profile_required_args():
    parser = _build_parser()
    args = parser.parse_args([
        "profile",
        "--model", "Qwen/Qwen3-4B",
        "--model-type", "qwen3",
        "--hardware", "RTX_4090",
    ])
    assert args.cmd == "profile"
    assert args.model == "Qwen/Qwen3-4B"
    assert args.model_type == "qwen3"
    assert args.hardware == "RTX_4090"
    assert args.dtype == "bfloat16"      # default
    assert args.tp == 1
    assert args.iterations == 3
    assert args.no_resume is False


def test_parser_profile_with_kinds():
    parser = _build_parser()
    args = parser.parse_args([
        "profile",
        "--model", "x", "--model-type", "qwen3", "--hardware", "RTX_4090",
        "--kinds", "dense", "per_sequence",
    ])
    assert args.kinds == ["dense", "per_sequence"]


def test_parser_profile_kinds_invalid_choice():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "profile",
            "--model", "x", "--model-type", "qwen3", "--hardware", "RTX_4090",
            "--kinds", "bogus",
        ])


def test_parser_fit_required_args():
    parser = _build_parser()
    args = parser.parse_args([
        "fit", "--raw", "/tmp/raw", "--out", "/tmp/efficiency.yaml",
    ])
    assert args.cmd == "fit"
    assert args.raw == "/tmp/raw"
    assert args.out == "/tmp/efficiency.yaml"


def test_main_profile_blocked_when_virtual_backend_set(monkeypatch):
    """VLLM_VIRTUAL_BACKEND=1 时 main 应返非零 + 提示."""
    monkeypatch.setenv("VLLM_VIRTUAL_BACKEND", "1")
    rc = main(["profile", "--model", "x", "--model-type", "qwen3",
               "--hardware", "RTX_4090"])
    assert rc == 2


def test_main_profile_calls_run_calibration(monkeypatch, tmp_path):
    """VLLM_VIRTUAL_BACKEND 未设时, main 应调 run_calibration."""
    monkeypatch.delenv("VLLM_VIRTUAL_BACKEND", raising=False)

    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)
        out = tmp_path / "out"
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr("llm_infer_sim.calibration.runner.run_calibration", fake_run)

    rc = main([
        "profile",
        "--model", "Qwen/Qwen3-4B",
        "--model-type", "qwen3",
        "--hardware", "RTX_4090",
        "--dtype", "bfloat16",
        "--tp", "2",
        "--iterations", "5",
        "--kinds", "dense",
    ])
    assert rc == 0
    assert called["model"] == "Qwen/Qwen3-4B"
    assert called["model_type"] == "qwen3"
    assert called["hardware"] == "RTX_4090"
    assert called["dtype"] == "bfloat16"
    assert called["tp"] == 2
    assert called["iterations"] == 5
    assert called["kinds"] == ("dense",)


def test_main_fit_missing_bundle_returns_2(tmp_path):
    """fit 子命令 (B.5): raw 目录没 bundle.yaml → 返 2 + 提示."""
    rc = main(["fit", "--raw", str(tmp_path), "--out", str(tmp_path / "y.yaml")])
    assert rc == 2


def test_main_no_subcommand():
    """无 subcommand → argparse error (SystemExit 2)."""
    with pytest.raises(SystemExit):
        main([])
