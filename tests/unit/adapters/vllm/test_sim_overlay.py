"""sim_overlay loader: 可选 YAML 薄覆盖层解析 + fail-fast 校验。"""
from __future__ import annotations

import pytest

from llm_infer_sim.adapters.vllm.sim_overlay import (
    SimOverlay,
    _clear_cache,
    load_sim_overlay,
)


def _write(tmp_path, body: str):
    _clear_cache()
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_missing_file_returns_empty_overlay(tmp_path):
    _clear_cache()
    overlay = load_sim_overlay(tmp_path / "absent.yaml")
    assert overlay == SimOverlay()
    # 所有字段为 None → 无覆盖意图
    assert overlay.hardware.name is None
    assert overlay.quantization.w_byte is None
    assert overlay.runtime.time_mode is None


def test_empty_file_returns_empty_overlay(tmp_path):
    p = _write(tmp_path, "")
    assert load_sim_overlay(p) == SimOverlay()


def test_full_yaml_maps_fields(tmp_path):
    p = _write(tmp_path, """
hardware:
  name: RTX_4090
  topology_hint: balanced
  mem_efficiency: 0.75
  compute_efficiency: 0.9
  comm_efficiency: 0.8
calibration:
  enabled: false
quantization:
  w_byte: auto
  a_byte: 1.0
  kv_byte: 2.0
pd_disagg:
  role: kv_both
  connector_name: P2pNcclConnector
  kv_parallel_size: 2
  connector_bandwidth_gbps: 25.0
  connector_latency_us: 5.0
runtime:
  time_mode: instant
  dump_ops: 2
  dump_requests: false
""")
    o = load_sim_overlay(p)
    assert o.hardware.name == "RTX_4090"
    assert o.hardware.topology_hint == "balanced"
    assert o.hardware.mem_efficiency == 0.75
    assert o.hardware.compute_efficiency == 0.9
    assert o.hardware.comm_efficiency == 0.8
    assert o.calibration.enabled is False
    # auto → None (沿用 vLLM 推导); 数值才覆盖
    assert o.quantization.w_byte is None
    assert o.quantization.a_byte == 1.0
    assert o.quantization.kv_byte == 2.0
    assert o.pd_disagg.role == "kv_both"
    assert o.pd_disagg.connector_name == "P2pNcclConnector"
    assert o.pd_disagg.kv_parallel_size == 2
    assert o.pd_disagg.connector_bandwidth_gbps == 25.0
    assert o.pd_disagg.connector_latency_us == 5.0
    assert o.runtime.time_mode == "instant"
    assert o.runtime.dump_ops == 2
    assert o.runtime.dump_requests == 0   # bool false → 0


def test_unknown_section_fails(tmp_path):
    p = _write(tmp_path, "operator_db:\n  enabled: true\n")
    with pytest.raises(ValueError, match="未知 section"):
        load_sim_overlay(p)


def test_unknown_key_fails(tmp_path):
    p = _write(tmp_path, "hardware:\n  naem: H100\n")
    with pytest.raises(ValueError, match="未知 key"):
        load_sim_overlay(p)


def test_bad_scalar_type_fails(tmp_path):
    p = _write(tmp_path, "hardware:\n  mem_efficiency: high\n")
    with pytest.raises(ValueError, match="mem_efficiency"):
        load_sim_overlay(p)


def test_bool_not_accepted_as_efficiency(tmp_path):
    # bool 是 int 子类, 但 efficiency 期望 number → 应拒绝。
    p = _write(tmp_path, "hardware:\n  mem_efficiency: true\n")
    with pytest.raises(ValueError, match="number"):
        load_sim_overlay(p)


def test_auto_rejected_for_non_quant_field(tmp_path):
    # auto 是 quant byte 专属哨兵; 用在 efficiency 这种 float 字段上应失败。
    p = _write(tmp_path, "hardware:\n  mem_efficiency: auto\n")
    with pytest.raises(ValueError, match="number"):
        load_sim_overlay(p)


def test_auto_accepted_only_for_quant_bytes(tmp_path):
    p = _write(tmp_path, "quantization:\n  w_byte: auto\n  a_byte: auto\n  kv_byte: auto\n")
    o = load_sim_overlay(p)
    assert o.quantization.w_byte is None
    assert o.quantization.a_byte is None
    assert o.quantization.kv_byte is None


def test_calibration_enabled_must_be_bool(tmp_path):
    p = _write(tmp_path, "calibration:\n  enabled: 1\n")
    with pytest.raises(ValueError, match="bool"):
        load_sim_overlay(p)


def test_top_level_must_be_mapping(tmp_path):
    p = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="顶层"):
        load_sim_overlay(p)


def test_null_value_treated_as_no_override(tmp_path):
    # YAML `key: null` 应视同未设, 不报错也不覆盖。
    p = _write(tmp_path, "pd_disagg:\n  role: null\n  connector_name: P2pNcclConnector\n")
    o = load_sim_overlay(p)
    assert o.pd_disagg.role is None
    assert o.pd_disagg.connector_name == "P2pNcclConnector"


def test_example_template_parses():
    from pathlib import Path
    _clear_cache()
    example = Path(__file__).resolve().parents[4] / "configs" / "config.yaml.example"
    assert example.exists(), example
    o = load_sim_overlay(example)
    assert o.hardware.name == "H100"
    assert o.runtime.time_mode == "realtime"
    assert o.quantization.w_byte is None   # auto


def test_process_cache_returns_same_instance(tmp_path):
    p = _write(tmp_path, "hardware:\n  name: H100\n")
    first = load_sim_overlay(p)
    second = load_sim_overlay(p)
    assert first is second
