"""EfficiencyProfile lookup table + YAML I/O (详设 §9.4.2 Plan B / B.0).

覆盖:
  1. EfficiencyEntry 字段
  2. lookup 优先级 (精确 > shape-wildcard > dtype-wildcard > op-only > fallback)
  3. fallback category 分流 (compute / mem / comm)
  4. placeholder() 向后兼容
  5. apply_to(hw) 把 default_* 应用到 HardwareConfig
  6. from_yaml / to_yaml round-trip
  7. YAML 缺字段 / 非法 entry 容错
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_infer_sim.core.profiles.efficiency_profile import (
    EfficiencyEntry,
    EfficiencyProfile,
)


# ---------- EfficiencyEntry ----------

def test_efficiency_entry_key():
    e = EfficiencyEntry(op_kind="dense_gemm", dtype="bf16", shape_key="tokens<=128",
                        efficiency=0.62)
    assert e.key() == ("dense_gemm", "bf16", "tokens<=128")


# ---------- placeholder backwards compat ----------

def test_placeholder_returns_all_ones():
    p = EfficiencyProfile.placeholder()
    assert p.default_compute == 1.0
    assert p.default_mem == 1.0
    assert p.default_comm == 1.0
    assert p.entries == {}
    assert p.w_byte == 2.0


def test_apply_to_hw_uses_defaults():
    p = EfficiencyProfile(default_compute=0.8, default_mem=0.7, default_comm=0.9)
    hw = SimpleNamespace(compute_efficiency=None, mem_efficiency=None,
                         comm_efficiency=None)
    p.apply_to(hw)
    assert hw.compute_efficiency == 0.8
    assert hw.mem_efficiency == 0.7
    assert hw.comm_efficiency == 0.9


# ---------- lookup ----------

def test_lookup_exact_match():
    p = EfficiencyProfile()
    p.add_entry(EfficiencyEntry("dense_gemm", "bf16", "tokens<=128", 0.62))
    assert p.lookup("dense_gemm", "bf16", "tokens<=128") == 0.62


def test_lookup_falls_back_to_default():
    p = EfficiencyProfile(default_compute=0.7)
    # entry 不存在
    assert p.lookup("dense_gemm", "bf16", "tokens<=128") == 0.7


def test_lookup_shape_wildcard():
    """精确 miss, shape="*" 命中."""
    p = EfficiencyProfile(default_compute=0.5)
    p.add_entry(EfficiencyEntry("dense_gemm", "bf16", "*", 0.80))
    assert p.lookup("dense_gemm", "bf16", "tokens<=128") == 0.80


def test_lookup_dtype_wildcard():
    """shape 命中 dtype="*"."""
    p = EfficiencyProfile()
    p.add_entry(EfficiencyEntry("rmsnorm", "*", "tokens<=128", 0.55))
    assert p.lookup("rmsnorm", "bf16", "tokens<=128") == 0.55
    assert p.lookup("rmsnorm", "fp8", "tokens<=128") == 0.55


def test_lookup_op_only_wildcard():
    """两个 * 都用."""
    p = EfficiencyProfile()
    p.add_entry(EfficiencyEntry("rope", "*", "*", 0.45))
    assert p.lookup("rope", "bf16", "tokens<=1024") == 0.45


def test_lookup_priority_exact_beats_wildcard():
    """精确 > shape-* > dtype-* > op-only."""
    p = EfficiencyProfile()
    p.add_entry(EfficiencyEntry("dense_gemm", "bf16", "tokens<=128", 0.90))
    p.add_entry(EfficiencyEntry("dense_gemm", "bf16", "*", 0.50))
    p.add_entry(EfficiencyEntry("dense_gemm", "*", "*", 0.30))
    assert p.lookup("dense_gemm", "bf16", "tokens<=128") == 0.90
    assert p.lookup("dense_gemm", "bf16", "tokens<=1024") == 0.50
    assert p.lookup("dense_gemm", "fp8", "tokens<=128") == 0.30


def test_lookup_category_routes_to_correct_default():
    p = EfficiencyProfile(default_compute=0.7, default_mem=0.85, default_comm=0.95)
    assert p.lookup("unknown", "bf16", "*", category="compute") == 0.7
    assert p.lookup("unknown", "bf16", "*", category="mem") == 0.85
    assert p.lookup("unknown", "bf16", "*", category="comm") == 0.95


# ---------- YAML roundtrip ----------

def _sample_profile() -> EfficiencyProfile:
    p = EfficiencyProfile(
        default_compute=0.7, default_mem=0.85, default_comm=1.0,
        hardware="RTX_4090", captured_at="2026-05-15", vllm_version="0.20.1",
    )
    p.add_entry(EfficiencyEntry("dense_gemm", "bf16", "tokens<=128", 0.62,
                                confidence=0.9, n_samples=8,
                                source="rtx_4090/Qwen3-4B/bf16"))
    p.add_entry(EfficiencyEntry("dense_gemm", "bf16", "tokens<=1024", 0.85,
                                n_samples=12))
    p.add_entry(EfficiencyEntry("rmsnorm", "bf16", "*", 0.55, n_samples=8))
    return p


def test_yaml_roundtrip_preserves_entries():
    p = _sample_profile()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.yaml"
        p.to_yaml(path)
        loaded = EfficiencyProfile.from_yaml(path)
        assert loaded.default_compute == p.default_compute
        assert loaded.default_mem == p.default_mem
        assert loaded.hardware == p.hardware
        assert loaded.captured_at == p.captured_at
        assert loaded.vllm_version == p.vllm_version
        assert len(loaded.entries) == len(p.entries)
        for k, v in p.entries.items():
            assert loaded.entries[k].efficiency == v.efficiency
            assert loaded.entries[k].n_samples == v.n_samples
            assert loaded.entries[k].source == v.source


def test_yaml_lookup_after_load_works():
    p = _sample_profile()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.yaml"
        p.to_yaml(path)
        loaded = EfficiencyProfile.from_yaml(path)
        assert loaded.lookup("dense_gemm", "bf16", "tokens<=128") == 0.62
        assert loaded.lookup("rmsnorm", "bf16", "tokens<=128") == 0.55  # shape wildcard


def test_from_yaml_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        EfficiencyProfile.from_yaml("/nonexistent/path/efficiency.yaml")


def test_from_yaml_minimal_uses_defaults():
    """YAML 只有 hardware 字段, 其他默认值 1.0/2.0."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "minimal.yaml"
        path.write_text("hardware: RTX_4090\n")
        p = EfficiencyProfile.from_yaml(path)
        assert p.hardware == "RTX_4090"
        assert p.default_compute == 1.0
        assert p.entries == {}


def test_from_yaml_skips_invalid_entry_with_warning():
    """损坏的 entry 应触发 warn 但不挂."""
    yaml_text = """
hardware: RTX_4090
entries:
  - {op_kind: dense_gemm, dtype: bf16, shape_key: "ok", efficiency: 0.5}
  - {op_kind: missing_required_field}
"""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "broken.yaml"
        path.write_text(yaml_text)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = EfficiencyProfile.from_yaml(path)
        # 良 entry 入表; 坏 entry skip + warn
        assert len(p.entries) == 1
        assert any("Skipping invalid" in str(x.message) for x in w)


def test_to_yaml_creates_parent_dir():
    """to_yaml 应能自动建 parent dir."""
    with tempfile.TemporaryDirectory() as td:
        nested = Path(td) / "nested" / "deeper" / "test.yaml"
        p = EfficiencyProfile.placeholder()
        p.to_yaml(nested)
        assert nested.exists()


def test_to_yaml_entries_sorted_for_diff_stability():
    """entries 按 key 排序写入, 让 git diff 友好."""
    p = EfficiencyProfile()
    p.add_entry(EfficiencyEntry("z_op", "bf16", "z", 0.1))
    p.add_entry(EfficiencyEntry("a_op", "bf16", "a", 0.9))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sorted.yaml"
        p.to_yaml(path)
        text = path.read_text()
        a_pos = text.index("a_op")
        z_pos = text.index("z_op")
        assert a_pos < z_pos
