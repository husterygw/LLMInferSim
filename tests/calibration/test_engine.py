"""calibration/engine.py — spin_up / spin_down / fire_shot (B.2).

整 vLLM 真启动需要 GPU + 模型, 单测里 mock 掉 vllm.LLM. 只验证 wiring 正确。
"""
from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

from llm_infer_sim.calibration.engine import spin_up, spin_down, fire_shot


def test_spin_up_raises_when_virtual_backend_set(monkeypatch):
    """VLLM_VIRTUAL_BACKEND=1 时拒跑 calibration."""
    monkeypatch.setenv("VLLM_VIRTUAL_BACKEND", "1")
    with pytest.raises(RuntimeError, match="VLLM_VIRTUAL_BACKEND=1"):
        spin_up(model="dummy")


def test_spin_up_invalid_dtype_raises(monkeypatch):
    monkeypatch.delenv("VLLM_VIRTUAL_BACKEND", raising=False)
    with pytest.raises(ValueError, match="dtype"):
        spin_up(model="dummy", dtype="int4")     # 不在 _VALID_DTYPES


def test_spin_up_constructs_LLM_with_correct_kwargs(monkeypatch):
    """patch vllm.LLM, 验证 spin_up 把正确的 worker_extension_cls + enforce_eager 等
    透传过去."""
    monkeypatch.delenv("VLLM_VIRTUAL_BACKEND", raising=False)

    captured_kwargs = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    fake_vllm = mock.MagicMock()
    fake_vllm.LLM = _FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    engine = spin_up(
        model="Qwen/Qwen3-4B", dtype="bfloat16", tp=1,
        max_model_len=4096, max_num_seqs=64,
    )
    assert isinstance(engine, _FakeLLM)
    assert captured_kwargs["model"] == "Qwen/Qwen3-4B"
    assert captured_kwargs["tensor_parallel_size"] == 1
    assert captured_kwargs["dtype"] == "bfloat16"
    assert captured_kwargs["enforce_eager"] is True
    assert captured_kwargs["max_logprobs"] == 0
    assert captured_kwargs["disable_log_stats"] is True
    assert captured_kwargs["worker_extension_cls"] == (
        "llm_infer_sim.calibration.extension.LayerwiseProfileExtension"
    )


def test_spin_up_extra_kwargs_merged(monkeypatch):
    monkeypatch.delenv("VLLM_VIRTUAL_BACKEND", raising=False)
    captured = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_vllm = mock.MagicMock()
    fake_vllm.LLM = _FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    spin_up(model="x", extra_engine_kwargs={"quantization": "fp8"})
    assert captured["quantization"] == "fp8"


def test_spin_down_clears_reference():
    """spin_down 没 raise (即使 engine 没 close 接口)."""
    engine = mock.MagicMock()
    spin_down(engine)  # 不该 raise


def test_fire_shot_uses_collective_rpc():
    """fire_shot 用 collective_rpc('fire', args=...) 透传."""
    engine = mock.MagicMock()
    engine.collective_rpc.return_value = [[{"layer": "qkv_proj", "op_kind": "dense_gemm",
                                            "microseconds": 100.0}]]
    shot = {"kind": "dense", "num_new_tokens": 128,
            "num_decode_seqs": 0, "kv_lens_prefill": [],
            "kv_lens_decode": [], "prefill_chunk": 0}
    slice_ = {"qkv_proj": {"vllm": "QKVParallelLinear",
                           "within": None, "op_kind": "dense_gemm"}}
    out = fire_shot(engine, shot, slice_, kind="dense", iterations=3)
    engine.collective_rpc.assert_called_once_with(
        "fire", args=(shot, slice_, "dense", 3),
    )
    assert out == [[{"layer": "qkv_proj", "op_kind": "dense_gemm", "microseconds": 100.0}]]
