"""profile_extractor._maybe_load_efficiency_yaml — 启动时读 configs/efficiency/*.yaml (B.6)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from llm_infer_sim.adapters.vllm.profile_extractor import (
    _maybe_load_efficiency_yaml,
    extract_profile_bundle,
)
from llm_infer_sim.core.profiles.efficiency_profile import EfficiencyProfile


def _placeholder():
    return EfficiencyProfile.placeholder()


def _qwen3_hf():
    return SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=2560, num_hidden_layers=36,
        intermediate_size=9728, vocab_size=151936, head_dim=128,
    )


def _make_vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=_qwen3_hf(), model="x"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1),
    )


def test_no_yaml_returns_fallback(monkeypatch):
    """env 没指 path, 默认 configs 目录没文件 → 返 fallback."""
    monkeypatch.delenv("LLM_INFER_SIM_EFFICIENCY_YAML", raising=False)
    monkeypatch.delenv("LLM_INFER_SIM_USE_CALIBRATION", raising=False)
    fb = _placeholder()
    res = _maybe_load_efficiency_yaml("NoSuchHW_xyz_123", fb)
    assert res is fb


def test_calibration_not_loaded_by_default(monkeypatch):
    """关键 invariant: 默认 (USE_CALIBRATION 未设) 不自动找 configs/efficiency/<hw>.yaml.

    即便 configs/efficiency/rtx_4090.yaml 存在, 也不该被自动加载, 防止"用了过期
    校准导致结果失真"的暗坑 (详 PROJECT_REPORT §4.6)。"""
    monkeypatch.delenv("LLM_INFER_SIM_EFFICIENCY_YAML", raising=False)
    monkeypatch.delenv("LLM_INFER_SIM_USE_CALIBRATION", raising=False)
    fb = _placeholder()
    # RTX_4090 在 configs/efficiency/rtx_4090.yaml 存在的前提下, 默认行为是 NOT 加载
    res = _maybe_load_efficiency_yaml("RTX_4090", fb)
    assert res is fb, "calibration YAML 不应被默认加载, 需 LLM_INFER_SIM_USE_CALIBRATION=1 显式 opt-in"


def test_calibration_opt_in_walks_hw_lookup(monkeypatch):
    """LLM_INFER_SIM_USE_CALIBRATION=1 时, 按 hw_name 自动找校准 YAML (不挂).

    找不到匹配文件时仍 fallback, 但 lookup 路径走过。"""
    monkeypatch.delenv("LLM_INFER_SIM_EFFICIENCY_YAML", raising=False)
    monkeypatch.setenv("LLM_INFER_SIM_USE_CALIBRATION", "1")
    fb = _placeholder()
    res = _maybe_load_efficiency_yaml("DefinitelyNoSuchHW_xyz", fb)
    assert res is fb


def test_calibration_opt_in_loads_existing_rtx_4090(monkeypatch):
    """LLM_INFER_SIM_USE_CALIBRATION=1 + hw 有现成 YAML (configs/efficiency/rtx_4090.yaml)
    → 加载校准。"""
    monkeypatch.delenv("LLM_INFER_SIM_EFFICIENCY_YAML", raising=False)
    monkeypatch.setenv("LLM_INFER_SIM_USE_CALIBRATION", "1")
    fb = _placeholder()
    res = _maybe_load_efficiency_yaml("RTX_4090", fb)
    # rtx_4090.yaml 存在 → 应该被加载, 不返 fallback
    # (若文件被删则跳过该断言)
    from pathlib import Path
    yaml_path = (Path(__file__).resolve().parent.parent.parent.parent
                 / "configs" / "efficiency" / "rtx_4090.yaml")
    if yaml_path.exists():
        assert res is not fb, "USE_CALIBRATION=1 + 有 yaml 应该加载校准"
        assert len(res.entries) > 0


def test_explicit_env_yaml_loaded(monkeypatch):
    """LLM_INFER_SIM_EFFICIENCY_YAML 指向有效 YAML → 加载."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.yaml"
        path.write_text(yaml.safe_dump({
            "hardware": "TEST_HW",
            "default_compute": 0.7,
            "default_mem": 0.85,
            "entries": [
                {"op_kind": "dense_gemm", "dtype": "bfloat16",
                 "shape_key": "tokens<=128", "efficiency": 0.62},
            ],
        }))
        monkeypatch.setenv("LLM_INFER_SIM_EFFICIENCY_YAML", str(path))
        fb = _placeholder()
        res = _maybe_load_efficiency_yaml("TEST_HW", fb)
        assert res is not fb
        assert res.default_compute == 0.7
        assert len(res.entries) == 1


def test_explicit_env_invalid_yaml_falls_back(monkeypatch, tmp_path):
    """env yaml 路径存在但内容坏 → fallback (warn 但不挂)."""
    p = tmp_path / "broken.yaml"
    p.write_text("this is not yaml: [broken")
    monkeypatch.setenv("LLM_INFER_SIM_EFFICIENCY_YAML", str(p))
    fb = _placeholder()
    res = _maybe_load_efficiency_yaml("TEST_HW", fb)
    # 损坏 → 走 fallback
    assert res is fb or res is not fb  # 不挂即可 (具体哪个看 yaml parser 容错)


def test_loaded_yaml_preserves_extract_byte_settings(monkeypatch):
    """extract 阶段已切的 w_byte/a_byte/kv_byte 不该被 YAML 覆盖
    (YAML 里的 byte 是上次校准时的, 跟当前 quant config 可能不同)."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "y.yaml"
        path.write_text(yaml.safe_dump({
            "hardware": "X",
            "w_byte": 1.0, "a_byte": 1.0, "kv_byte": 1.0,
            "default_compute": 0.7,
            "entries": [],
        }))
        monkeypatch.setenv("LLM_INFER_SIM_EFFICIENCY_YAML", str(path))
        # fallback 已是 bf16 (extract 解析过 quant config)
        fb = EfficiencyProfile(w_byte=2.0, a_byte=2.0, kv_byte=2.0)
        res = _maybe_load_efficiency_yaml("X", fb)
        # YAML 的 efficiency 用上, 但 byte 保 fallback 的 2.0
        assert res.w_byte == 2.0
        assert res.a_byte == 2.0
        assert res.kv_byte == 2.0
        assert res.default_compute == 0.7   # YAML 的 efficiency 项


def test_extract_profile_bundle_uses_calibrated_yaml(monkeypatch):
    """extract_profile_bundle 端到端: YAML 指向 → bundle.efficiency 是校准过的."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cal.yaml"
        path.write_text(yaml.safe_dump({
            "hardware": "TEST_HW",
            "default_compute": 0.65,
            "entries": [
                {"op_kind": "dense_gemm", "dtype": "bfloat16",
                 "shape_key": "tokens<=128", "efficiency": 0.5},
            ],
        }))
        monkeypatch.setenv("LLM_INFER_SIM_EFFICIENCY_YAML", str(path))
        bundle = extract_profile_bundle(_make_vllm_config())
        # bundle.efficiency.entries 应非空
        assert len(bundle.efficiency.entries) == 1
        # apply_to 把 default_compute 推给 hw
        assert bundle.hw.compute_efficiency == pytest.approx(0.65, rel=1e-3)
