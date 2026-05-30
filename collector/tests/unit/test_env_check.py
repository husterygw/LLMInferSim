"""env_check.py — 环境快照采集."""
from __future__ import annotations

from collector import env_check


def test_collect_env_returns_snapshot():
    """不 raise, 即使 GPU 不在 / nvidia-smi 不在."""
    snap = env_check.collect_env(collector_version="0.1.0")
    assert snap.collector_version == "0.1.0"
    assert snap.python_version    # 至少 python 版本能拿到
    assert snap.captured_at


def test_collect_env_torch_version_detected():
    snap = env_check.collect_env()
    # torch 在 test env 应该装着 (vllm 依赖)
    assert snap.torch_version != ""


def test_auto_hardware_id_4090():
    snap = env_check.EnvSnapshot(gpu_name="NVIDIA GeForce RTX 4090")
    assert env_check.auto_hardware_id(snap) == "RTX_4090"


def test_auto_hardware_id_h100():
    snap = env_check.EnvSnapshot(gpu_name="NVIDIA H100 PCIe")
    assert env_check.auto_hardware_id(snap) == "H100"


def test_auto_hardware_id_unknown_fallback():
    snap = env_check.EnvSnapshot(gpu_name="some weird GPU name")
    h = env_check.auto_hardware_id(snap)
    assert "_" in h or h == "unknown"   # 空格换 _


def test_auto_hardware_id_empty():
    snap = env_check.EnvSnapshot(gpu_name="")
    assert env_check.auto_hardware_id(snap) == "unknown"


def test_write_manifest(tmp_path):
    """写出 yaml 文件可读回."""
    snap = env_check.EnvSnapshot(
        gpu_name="RTX 4090",
        gpu_count=8,
        driver_version="550.54.15",
        vllm_version="0.19.1",
        captured_at="2026-05-19T00:00:00",
    )
    path = tmp_path / "manifest.yaml"
    env_check.write_manifest(snap, path)
    assert path.exists()
    content = path.read_text()
    assert "RTX 4090" in content
    assert "550.54.15" in content


def test_unlocked_gpu_warning(monkeypatch):
    """lock_freq=None + gpu_count > 0 → 警告写到 warnings."""
    # mock _detect_* 让 GPU 存在但未锁
    monkeypatch.setattr(env_check, "_detect_gpu_count", lambda: 1)
    monkeypatch.setattr(env_check, "_detect_gpu_freq_lock", lambda: None)
    monkeypatch.setattr(env_check, "_detect_compute_mode", lambda: "Default")
    monkeypatch.setattr(env_check, "_detect_gpu_name", lambda: "RTX 4090")

    snap = env_check.collect_env()
    assert any("频率未锁" in w for w in snap.warnings)
    assert any("Default" in w for w in snap.warnings)


def test_no_gpu_no_warnings(monkeypatch):
    """gpu_count=0 时不该 warn 锁频."""
    monkeypatch.setattr(env_check, "_detect_gpu_count", lambda: 0)
    monkeypatch.setattr(env_check, "_detect_gpu_freq_lock", lambda: None)
    snap = env_check.collect_env()
    # 不该有 "频率未锁" 的 warn
    assert not any("频率未锁" in w for w in snap.warnings)


def test_warn_disabled(monkeypatch):
    monkeypatch.setattr(env_check, "_detect_gpu_count", lambda: 1)
    monkeypatch.setattr(env_check, "_detect_gpu_freq_lock", lambda: None)
    snap = env_check.collect_env(warn_unlocked_gpu=False)
    assert not any("频率未锁" in w for w in snap.warnings)
