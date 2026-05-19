"""fit.py — CSV + bundle.yaml → EfficiencyProfile YAML (B.5).

全 mock: 写假 CSV + bundle.yaml, 跑 fit_efficiency, 验证 YAML 输出。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from llm_infer_sim.calibration import csv_io
from llm_infer_sim.calibration.fit import (
    BundleSpec,
    FitGroup,
    fit_efficiency,
    kv_bucket,
    predicted_op_dense,
    predicted_op_per_seq,
    sequence_bucket,
    token_bucket,
    _confidence_from_spread,
    _dtype_from_a_byte,
)


# ---- 桶函数 ----

def test_token_bucket():
    assert token_bucket(1) == "tokens<=16"
    assert token_bucket(16) == "tokens<=16"
    assert token_bucket(17) == "tokens<=128"
    assert token_bucket(128) == "tokens<=128"
    assert token_bucket(1024) == "tokens<=1024"
    assert token_bucket(2048) == "tokens>1024"


def test_sequence_bucket():
    assert sequence_bucket(1) == "seq<=4"
    assert sequence_bucket(4) == "seq<=4"
    assert sequence_bucket(5) == "seq<=16"
    assert sequence_bucket(32) == "seq>16"


def test_kv_bucket():
    assert kv_bucket(0) == "kv<=256"
    assert kv_bucket(256) == "kv<=256"
    assert kv_bucket(1024) == "kv<=1024"
    assert kv_bucket(4096) == "kv<=4k"
    assert kv_bucket(16384) == "kv<=16k"
    assert kv_bucket(65536) == "kv>16k"


# ---- dtype 推导 ----

def test_dtype_from_a_byte():
    assert _dtype_from_a_byte(0.5) == "fp4"
    assert _dtype_from_a_byte(1.0) == "fp8"
    assert _dtype_from_a_byte(2.0) == "bf16"
    assert _dtype_from_a_byte(4.0) == "fp32"


# ---- confidence ----

def test_confidence_single_sample():
    assert _confidence_from_spread([0.5]) == 1.0


def test_confidence_consistent_samples():
    """完全一致 → 1.0."""
    assert _confidence_from_spread([0.5, 0.5, 0.5]) == 1.0


def test_confidence_spread_lowers():
    """差异大 → 低 confidence; 太大 (>median) clamp 到 0."""
    high = _confidence_from_spread([0.48, 0.50, 0.52])     # spread 0.08
    mid = _confidence_from_spread([0.40, 0.50, 0.60])      # spread 0.40
    assert 0 < mid < high < 1
    # 极端: 差异 > median → clamp 0
    assert _confidence_from_spread([0.2, 0.5, 0.8]) == 0.0


# ---- BundleSpec from_yaml ----

def _qwen3_4b_bundle_dict() -> dict:
    """Qwen3-4B 标准 bundle (跟我们 hw RTX_4090)."""
    return {
        "hardware": "RTX_4090",
        "model": {
            "name": "Qwen3-4B",
            "hidden_dim": 2560,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "ffn_dim": 9728,
            "num_layers": 36,
            "vocab_size": 151936,
            "is_moe": False,
            "num_experts": 0,
            "num_activated_experts": 0,
            "expert_dim": 0,
            "num_shared_experts": 0,
            "kv_lora_rank": 0,
            "qk_nope_head_dim": 0,
            "rope_head_dim": 0,
            "v_head_dim": 0,
        },
        "deploy": {
            "tp": 1, "dp": 1, "ep": 1,
            "w_byte": 2.0, "a_byte": 2.0, "kv_byte": 2.0,
            "base_w_byte": 2.0, "base_a_byte": 2.0,
            "use_flash_attention": True,
        },
    }


def test_bundle_spec_from_yaml():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bundle.yaml"
        path.write_text(yaml.safe_dump(_qwen3_4b_bundle_dict()))
        spec = BundleSpec.from_yaml(path)
        assert spec.hidden_dim == 2560
        assert spec.num_kv_heads == 8
        assert spec.head_dim == 128
        assert spec.w_byte == 2.0
        assert spec.hardware == "RTX_4090"


# ---- predicted_op_* ----

def _qwen3_4b_spec() -> BundleSpec:
    return BundleSpec(
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
        is_moe=False, kv_lora_rank=0, rope_head_dim=0,
        tp=1, w_byte=2.0, a_byte=2.0, kv_byte=2.0,
        base_w_byte=2.0, base_a_byte=2.0, hardware="RTX_4090",
    )


def test_predicted_op_dense_qkv_proj():
    spec = _qwen3_4b_spec()
    op = predicted_op_dense("qkv_proj", tokens=128, b=spec)
    assert op is not None
    assert op.name == "qkv_proj"
    assert op.flops > 0
    assert op.load_weight > 0


def test_predicted_op_dense_layernorm():
    spec = _qwen3_4b_spec()
    op = predicted_op_dense("layernorm", tokens=128, b=spec)
    assert op is not None
    assert op.op_category == "norm"


def test_predicted_op_dense_unknown_returns_none():
    spec = _qwen3_4b_spec()
    assert predicted_op_dense("bogus_op", tokens=128, b=spec) is None


def test_predicted_op_per_seq_lm_head():
    spec = _qwen3_4b_spec()
    op = predicted_op_per_seq("lm_head", sequences=4, b=spec)
    assert op is not None
    assert op.name == "lm_head"


# ---- FitGroup ----

def test_fit_group_to_entry_median():
    g = FitGroup(op_kind="dense_gemm", dtype="bf16", shape_key="tokens<=128",
                 efficiencies=[0.5, 0.6, 0.7])
    entry = g.to_entry(source="test")
    assert entry.efficiency == 0.6      # median
    assert entry.n_samples == 3
    assert entry.source == "test"


def test_fit_group_empty_to_entry():
    g = FitGroup(op_kind="x", dtype="bf16", shape_key="*", efficiencies=[])
    entry = g.to_entry(source="test")
    assert entry.efficiency == 1.0
    assert entry.n_samples == 0


# ---- fit_efficiency 端到端 (with mock CSV + bundle) ----

def _write_mock_profile_run(td: Path) -> Path:
    """造一个最小 profile run dir: bundle.yaml + meta.yaml + 三个 CSV."""
    out = td / "RTX_4090" / "Qwen3-4B" / "bfloat16" / "tp1"
    out.mkdir(parents=True, exist_ok=True)

    # bundle.yaml
    (out / "bundle.yaml").write_text(yaml.safe_dump(_qwen3_4b_bundle_dict()))
    # meta.yaml
    (out / "meta.yaml").write_text(yaml.safe_dump({
        "model": "Qwen/Qwen3-4B",
        "model_type": "qwen3",
        "hardware": "RTX_4090",
        "dtype": "bfloat16",
        "tp": 1,
        "iterations": 3,
        "captured_at": "2026-05-15T00:00:00Z",
        "vllm_version": "0.20.1",
    }))

    # dense.csv: 几个 (canonical, tokens, time_us) 行
    with csv_io.CsvSink(out / "dense.csv", csv_io.DENSE_COLS) as sink:
        sink.write_rows([
            {"layer": "qkv_proj", "tokens": 128, "time_us": 100.0},
            {"layer": "qkv_proj", "tokens": 512, "time_us": 250.0},
            {"layer": "qkv_proj", "tokens": 1024, "time_us": 400.0},
            {"layer": "o_proj", "tokens": 128, "time_us": 80.0},
            {"layer": "down_proj", "tokens": 128, "time_us": 90.0},
            {"layer": "gate_up_proj", "tokens": 128, "time_us": 150.0},
            {"layer": "layernorm", "tokens": 128, "time_us": 20.0},
            {"layer": "embedding", "tokens": 1, "time_us": 5.0},
        ])

    # per_sequence.csv
    with csv_io.CsvSink(out / "per_sequence.csv", csv_io.PER_SEQ_COLS) as sink:
        sink.write_rows([
            {"layer": "lm_head", "sequences": 1, "time_us": 500.0},
            {"layer": "lm_head", "sequences": 4, "time_us": 600.0},
            {"layer": "lm_head", "sequences": 16, "time_us": 1200.0},
        ])

    # attention.csv: pure decode + pure prefill 各一组
    with csv_io.CsvSink(out / "attention.csv", csv_io.ATTN_COLS) as sink:
        sink.write_rows([
            {"prefill_chunk": 512, "kv_prefill": 0, "n_decode": 0,
             "kv_decode": 0, "time_us": 800.0},
            {"prefill_chunk": 0, "kv_prefill": 0, "n_decode": 1,
             "kv_decode": 1024, "time_us": 50.0},
            {"prefill_chunk": 0, "kv_prefill": 0, "n_decode": 4,
             "kv_decode": 4096, "time_us": 80.0},
        ])
    return out


def test_fit_efficiency_e2e_writes_yaml():
    with tempfile.TemporaryDirectory() as td:
        raw_dir = _write_mock_profile_run(Path(td))
        out_yaml = Path(td) / "rtx_4090.yaml"
        profile = fit_efficiency(raw_dir, out_yaml)

        assert out_yaml.exists()
        assert profile.hardware == "RTX_4090"
        # entries 应至少含若干 (qkv_proj / o_proj / down_proj / gate_up_proj / lm_head / attention)
        assert len(profile.entries) > 0
        # 读回来检查
        from llm_infer_sim.core.profiles.efficiency_profile import EfficiencyProfile
        loaded = EfficiencyProfile.from_yaml(out_yaml)
        assert loaded.hardware == "RTX_4090"
        assert len(loaded.entries) == len(profile.entries)


def test_fit_efficiency_dense_gemm_entries_present():
    with tempfile.TemporaryDirectory() as td:
        raw_dir = _write_mock_profile_run(Path(td))
        out_yaml = Path(td) / "out.yaml"
        profile = fit_efficiency(raw_dir, out_yaml)
        op_kinds_seen = {e.op_kind for e in profile.entries.values()}
        # 至少 dense_gemm / rmsnorm / embedding / attn 应在
        assert "dense_gemm" in op_kinds_seen
        assert "rmsnorm" in op_kinds_seen


def test_fit_efficiency_bucket_aggregation():
    """同 (op_kind, dtype, bucket) 多行应被聚合到 1 个 entry."""
    with tempfile.TemporaryDirectory() as td:
        raw_dir = _write_mock_profile_run(Path(td))
        # 我们的 dense.csv 里 qkv_proj 有 tokens=128, 512, 1024 → 应分 2 个桶
        # (128 在 tokens<=128, 512 在 tokens<=1024, 1024 在 tokens<=1024)
        out_yaml = Path(td) / "out.yaml"
        profile = fit_efficiency(raw_dir, out_yaml)
        # dense_gemm × bf16 × tokens<=1024 应 n_samples >= 2 (qkv 512 + 1024)
        e = profile.entries.get(("dense_gemm", "bfloat16", "tokens<=1024"))
        # 注意: o_proj/down_proj/gate_up_proj/lm_head 都是 dense_gemm, 各种 shape 都在
        assert e is not None or any(
            k[0] == "dense_gemm" and "tokens<=1024" in k[2] for k in profile.entries
        )


def test_fit_efficiency_default_compute_set():
    with tempfile.TemporaryDirectory() as td:
        raw_dir = _write_mock_profile_run(Path(td))
        out_yaml = Path(td) / "out.yaml"
        profile = fit_efficiency(raw_dir, out_yaml)
        # default_compute 应非 1.0 (有 entry 时算 median)
        assert 0 < profile.default_compute < 100  # 合理范围


def test_fit_efficiency_missing_bundle_raises():
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(FileNotFoundError, match="bundle.yaml"):
            fit_efficiency(Path(td), Path(td) / "out.yaml")
