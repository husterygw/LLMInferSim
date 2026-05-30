"""阶段 0 退出条件: vLLM platform plugin 注册 + check_and_update_config 注入 worker_cls。

这一组测试是 "pure pytest, 不实例化 LLM" 设计:
  - importlib.metadata 静态查 entry_point (无任何 vllm import)
  - virtual_platform_plugin() 函数行为 (会触发 import VirtualPlatform → import vllm,
    但只是 import 不实例化)
  - VirtualPlatform.check_and_update_config 用 mock VllmConfig 直接调

启动真实 vLLM LLM 实例的 e2e 验证在 examples/vllm_virtual/run_platform_selected.py。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------- entry_point 注册


def test_entry_point_registered_in_metadata():
    """importlib.metadata 能找到 'virtual' entry_point。"""
    from importlib.metadata import entry_points
    eps = entry_points(group="vllm.platform_plugins")
    names = sorted(ep.name for ep in eps)
    assert "virtual" in names, f"virtual not in {names}"


def test_entry_point_value_correct():
    """entry_point 的值指向我们的 plugin function qualname。"""
    from importlib.metadata import entry_points
    ep = next(
        ep for ep in entry_points(group="vllm.platform_plugins")
        if ep.name == "virtual"
    )
    # value 形如 "llm_infer_sim.adapters.vllm.virtual_platform:virtual_platform_plugin"
    assert ep.value == (
        "llm_infer_sim.adapters.vllm.virtual_platform:virtual_platform_plugin"
    )


# ---------------------------------------------------------------- plugin function


def test_plugin_returns_qualname_when_enabled(monkeypatch):
    """VLLM_VIRTUAL_BACKEND=1 时, plugin function 返回 VirtualPlatform qualname。"""
    monkeypatch.setenv("VLLM_VIRTUAL_BACKEND", "1")
    from llm_infer_sim.adapters.vllm.virtual_platform import virtual_platform_plugin
    qualname = virtual_platform_plugin()
    assert qualname == "llm_infer_sim.adapters.vllm.virtual_platform.VirtualPlatform"


def test_plugin_returns_none_when_disabled(monkeypatch):
    """VLLM_VIRTUAL_BACKEND 未设 / =0 时, plugin function 返回 None
    (让 vllm 走默认平台探测路径, 装了本包但没启用时不影响其他用法)。"""
    monkeypatch.delenv("VLLM_VIRTUAL_BACKEND", raising=False)
    from llm_infer_sim.adapters.vllm.virtual_platform import virtual_platform_plugin
    assert virtual_platform_plugin() is None

    monkeypatch.setenv("VLLM_VIRTUAL_BACKEND", "0")
    assert virtual_platform_plugin() is None


# ---------------------------------------------------------------- check_and_update_config


def _make_mock_vllm_config(worker_cls="auto"):
    """合成最小 VllmConfig 形态供 check_and_update_config 调用。

    阶段 3 起含 6 类 feature gate 字段, 全部置为干净默认 (不触发 fail-fast)。
    """
    return SimpleNamespace(
        parallel_config=SimpleNamespace(worker_cls=worker_cls),
        load_config=SimpleNamespace(load_format="auto"),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4]),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        # 阶段 3 C 块: feature gate 必需字段 (干净默认, 不应触发 raise)
        lora_config=None,
        speculative_config=None,
        decoding_config=SimpleNamespace(guided_decoding_backend=None),
        kv_transfer_config=None,
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            max_logprobs=0,
        ),
    )


def test_check_and_update_config_injects_worker_cls():
    """worker_cls = 'auto' 时, check_and_update_config 注入 VirtualWorker 路径。"""
    from llm_infer_sim.adapters.vllm.virtual_platform import VirtualPlatform
    cfg = _make_mock_vllm_config(worker_cls="auto")
    VirtualPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls == (
        "llm_infer_sim.adapters.vllm.virtual_worker.VirtualWorker"
    )


def test_check_and_update_config_preserves_explicit_worker_cls():
    """如果用户显式指定了 worker_cls, 不覆盖。"""
    from llm_infer_sim.adapters.vllm.virtual_platform import VirtualPlatform
    cfg = _make_mock_vllm_config(worker_cls="custom.WorkerCls")
    VirtualPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls == "custom.WorkerCls"


def test_check_and_update_config_forces_load_format_dummy():
    """load_format 强制 = 'dummy' 防御性拦截真实权重加载路径。"""
    from llm_infer_sim.adapters.vllm.virtual_platform import VirtualPlatform
    cfg = _make_mock_vllm_config()
    VirtualPlatform.check_and_update_config(cfg)
    assert cfg.load_config.load_format == "dummy"


def test_check_and_update_config_clears_cudagraph_capture_sizes():
    """cudagraph_capture_sizes 强制清空 (没真实模型, 不可能 capture)。"""
    from llm_infer_sim.adapters.vllm.virtual_platform import VirtualPlatform
    cfg = _make_mock_vllm_config()
    assert cfg.compilation_config.cudagraph_capture_sizes  # before: 非空
    VirtualPlatform.check_and_update_config(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes == []


def test_check_and_update_config_disables_async_scheduling():
    """async_scheduling 强制关 (与 sleep-based virtual barrier 假设不一致)。"""
    from llm_infer_sim.adapters.vllm.virtual_platform import VirtualPlatform
    cfg = _make_mock_vllm_config()
    assert cfg.scheduler_config.async_scheduling is True
    VirtualPlatform.check_and_update_config(cfg)
    assert cfg.scheduler_config.async_scheduling is False


# ---------------------------------------------------------------- platform attrs


def test_virtual_platform_attrs_compatible_with_vllm():
    """VirtualPlatform 的几个 attr 必须满足 vLLM 内部硬约束。"""
    from llm_infer_sim.adapters.vllm.virtual_platform import VirtualPlatform
    # device_name 必须是 torch 接受的 device 类型字符串 —— vLLM GroupCoordinator
    # 会做 torch.device(f"{device_name}:{rank}"), 我们故意声称 cpu (底层确实是 CPU tensor)
    assert VirtualPlatform.device_name == "cpu"
    # gloo backend 用于 CPU-only 多 worker
    assert VirtualPlatform.dist_backend == "gloo"
    # 简单 compile 后端避开 torch.compile / cudagraph
    assert VirtualPlatform.simple_compile_backend == "eager"


def test_virtual_platform_get_attn_backend_returns_empty():
    """get_attn_backend_cls 返回空字符串 → vLLM 跳过真实 attention backend 加载。"""
    from llm_infer_sim.adapters.vllm.virtual_platform import VirtualPlatform
    assert VirtualPlatform.get_attn_backend_cls() == ""


# ---------------------------------------------------------------- pyproject deps


def test_pyproject_pins_vllm_version():
    """pyproject.toml 锁 vllm==0.20.1 (阶段 0 spike 选定的版本)."""
    import tomllib
    from pathlib import Path
    pyproject = Path(__file__).parents[3] / "pyproject.toml"
    with pyproject.open("rb") as f:
        meta = tomllib.load(f)
    deps = meta["project"]["dependencies"]
    assert any("vllm==0.20.1" in d for d in deps), f"vllm not pinned in {deps}"
