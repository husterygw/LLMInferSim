"""overlay 接入 profile_extractor / virtual_model_runner: 优先级 vLLM < config.yaml < env。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_infer_sim.adapters.vllm import profile_extractor as pe
from llm_infer_sim.adapters.vllm import virtual_model_runner as vmr
from llm_infer_sim.adapters.vllm.profile_extractor import extract_scenario
from llm_infer_sim.adapters.vllm.sim_overlay import (
    CalibrationOverlay,
    HardwareOverlay,
    PDDisaggOverlay,
    QuantizationOverlay,
    RuntimeOverlay,
    SimOverlay,
)


def _qwen3_4b_vc():
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=2560, num_hidden_layers=36,
        intermediate_size=9728, vocab_size=151936, head_dim=128,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="Qwen/Qwen3-4B"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=64),
    )


def _patch_overlay(monkeypatch, overlay, target=pe):
    monkeypatch.setattr(target, "load_sim_overlay", lambda *a, **k: overlay)


def _clear_pd_env(monkeypatch):
    for k in (
        "LLM_INFER_SIM_HW", "LLM_INFER_SIM_MEM_EFFICIENCY", "LLM_INFER_SIM_NUMA_HINT",
        "LLM_INFER_SIM_PD_ROLE",
    ):
        monkeypatch.delenv(k, raising=False)


# ---- hardware ----

def test_yaml_hardware_name_selects_hw(monkeypatch):
    _clear_pd_env(monkeypatch)
    _patch_overlay(monkeypatch, SimOverlay(hardware=HardwareOverlay(name="RTX_4090")))
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.hardware.name == "RTX_4090"


def test_env_hw_wins_over_yaml(monkeypatch):
    _clear_pd_env(monkeypatch)
    monkeypatch.setenv("LLM_INFER_SIM_HW", "H100")
    _patch_overlay(monkeypatch, SimOverlay(hardware=HardwareOverlay(name="RTX_4090")))
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.hardware.name == "H100"


def test_default_hw_when_no_yaml_no_env(monkeypatch):
    _clear_pd_env(monkeypatch)
    _patch_overlay(monkeypatch, SimOverlay())
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.hardware.name == "H100"


def test_yaml_mem_efficiency_applied(monkeypatch):
    _clear_pd_env(monkeypatch)
    _patch_overlay(monkeypatch, SimOverlay(hardware=HardwareOverlay(mem_efficiency=0.75)))
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.hardware.to_legacy().mem_efficiency == 0.75


def test_env_mem_efficiency_wins_over_yaml(monkeypatch):
    _clear_pd_env(monkeypatch)
    monkeypatch.setenv("LLM_INFER_SIM_MEM_EFFICIENCY", "0.5")
    _patch_overlay(monkeypatch, SimOverlay(hardware=HardwareOverlay(mem_efficiency=0.75)))
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.hardware.to_legacy().mem_efficiency == 0.5


def test_yaml_topology_hint_applied(monkeypatch):
    _clear_pd_env(monkeypatch)
    _patch_overlay(monkeypatch, SimOverlay(hardware=HardwareOverlay(topology_hint="balanced")))
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.runtime.kernels.topology_hint == "balanced"


def test_env_numa_hint_wins_over_yaml(monkeypatch):
    _clear_pd_env(monkeypatch)
    monkeypatch.setenv("LLM_INFER_SIM_NUMA_HINT", "concentrated")
    _patch_overlay(monkeypatch, SimOverlay(hardware=HardwareOverlay(topology_hint="balanced")))
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.runtime.kernels.topology_hint == "concentrated"


# ---- quantization ----

def test_yaml_quant_override_only_when_not_auto(monkeypatch):
    _clear_pd_env(monkeypatch)
    # w_byte=None (auto) → 沿用 vLLM 推导 2.0; a_byte=1.0 覆盖。
    _patch_overlay(monkeypatch, SimOverlay(quantization=QuantizationOverlay(a_byte=1.0)))
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.model.quantization.w_byte == 2.0   # auto → vLLM-derived
    assert scenario.model.quantization.a_byte == 1.0   # overridden


# ---- calibration ----

def test_yaml_calibration_disabled_returns_placeholder(monkeypatch):
    _clear_pd_env(monkeypatch)
    # RTX_4090 calibrated 时有 moe_efficiency; disabled 时应为 placeholder (None)。
    _patch_overlay(monkeypatch, SimOverlay(
        hardware=HardwareOverlay(name="RTX_4090"),
        calibration=CalibrationOverlay(enabled=False),
    ))
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.calibration.moe_efficiency is None


def test_calibration_enabled_default_keeps_rtx4090_profile(monkeypatch):
    _clear_pd_env(monkeypatch)
    _patch_overlay(monkeypatch, SimOverlay(hardware=HardwareOverlay(name="RTX_4090")))
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.calibration.moe_efficiency is not None


# ---- pd_disagg ----

def test_yaml_pd_fields_override(monkeypatch):
    _clear_pd_env(monkeypatch)
    _patch_overlay(monkeypatch, SimOverlay(pd_disagg=PDDisaggOverlay(
        role="kv_both", connector_name="P2pNcclConnector",
        connector_bandwidth_gbps=25.0, connector_latency_us=5.0,
    )))
    scenario = extract_scenario(_qwen3_4b_vc())
    pd = scenario.deployment.pd
    assert pd.role == "kv_both"
    assert pd.connector_name == "P2pNcclConnector"
    assert pd.connector_bandwidth_gbps == 25.0
    assert pd.connector_latency_us == 5.0


def test_env_pd_role_wins_over_yaml(monkeypatch):
    _clear_pd_env(monkeypatch)
    monkeypatch.setenv("LLM_INFER_SIM_PD_ROLE", "kv_producer")
    _patch_overlay(monkeypatch, SimOverlay(pd_disagg=PDDisaggOverlay(role="kv_both")))
    scenario = extract_scenario(_qwen3_4b_vc())
    assert scenario.deployment.pd.role == "kv_producer"


# ---- runtime overlay in VirtualModelRunner ----

def _runner_runtime_env_keys():
    return ("LLM_INFER_SIM_TIME_MODE", "LLM_INFER_SIM_DUMP_OPS",
            "LLM_INFER_SIM_DUMP_REQUESTS")


def test_runner_yaml_runtime_applied(monkeypatch):
    _clear_pd_env(monkeypatch)
    for k in _runner_runtime_env_keys():
        monkeypatch.delenv(k, raising=False)
    # extractor 不受 overlay 影响 (空), runner 用 runtime overlay
    _patch_overlay(monkeypatch, SimOverlay(), target=pe)
    _patch_overlay(
        monkeypatch,
        SimOverlay(runtime=RuntimeOverlay(time_mode="instant", dump_ops=2, dump_requests=1)),
        target=vmr,
    )
    runner = vmr.VirtualModelRunner(_qwen3_4b_vc())
    assert runner.time_emulator.mode == "instant"
    assert runner._dump_ops_mode == 2
    assert runner._dump_requests_mode == 1


def test_runner_env_wins_over_yaml_runtime(monkeypatch):
    _clear_pd_env(monkeypatch)
    monkeypatch.setenv("LLM_INFER_SIM_TIME_MODE", "realtime")
    monkeypatch.setenv("LLM_INFER_SIM_DUMP_OPS", "0")
    monkeypatch.delenv("LLM_INFER_SIM_DUMP_REQUESTS", raising=False)
    _patch_overlay(monkeypatch, SimOverlay(), target=pe)
    _patch_overlay(
        monkeypatch,
        SimOverlay(runtime=RuntimeOverlay(time_mode="instant", dump_ops=2, dump_requests=1)),
        target=vmr,
    )
    runner = vmr.VirtualModelRunner(_qwen3_4b_vc())
    assert runner.time_emulator.mode == "realtime"   # env wins
    assert runner._dump_ops_mode == 0                 # env wins
    assert runner._dump_requests_mode == 1            # no env → YAML
