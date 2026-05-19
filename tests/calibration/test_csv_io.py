"""csv_io.py — 三类 CSV 读写 + sample→row 转换 (B.3)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from llm_infer_sim.calibration.csv_io import (
    CsvSink,
    AttnRow,
    DenseRow,
    PerSeqRow,
    DENSE_COLS,
    ATTN_COLS,
    PER_SEQ_COLS,
    read_dense,
    read_per_sequence,
    read_attention,
    samples_to_dense_rows,
    samples_to_per_seq_rows,
    samples_to_attn_rows,
    visited_keys_dense,
    visited_keys_per_sequence,
    visited_keys_attention,
)
from llm_infer_sim.calibration.shots import Shot
from llm_infer_sim.calibration.timings import TimingSample


# ---- 列 schema 对齐 LLMServingSim ----

def test_dense_cols_match_servingsim():
    assert DENSE_COLS == ("layer", "tokens", "time_us")


def test_per_seq_cols():
    assert PER_SEQ_COLS == ("layer", "sequences", "time_us")


def test_attn_cols():
    assert ATTN_COLS == ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us")


# ---- CsvSink 写入 + 读回 ----

def test_sink_writes_header_then_rows():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        with CsvSink(p, DENSE_COLS) as sink:
            sink.write_rows([{"layer": "qkv_proj", "tokens": 128, "time_us": 42.0}])
        text = p.read_text()
        assert text.startswith("layer,tokens,time_us\n")
        assert "qkv_proj,128,42.0" in text


def test_sink_appends_without_rewriting_header():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        with CsvSink(p, DENSE_COLS) as sink:
            sink.write_rows([{"layer": "a", "tokens": 1, "time_us": 1.0}])
        with CsvSink(p, DENSE_COLS) as sink:
            sink.write_rows([{"layer": "b", "tokens": 2, "time_us": 2.0}])
        text = p.read_text()
        # header 只出现一次
        assert text.count("layer,tokens,time_us") == 1
        assert "a,1,1.0" in text and "b,2,2.0" in text


def test_sink_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "deep" / "nested" / "out.csv"
        with CsvSink(p, DENSE_COLS) as sink:
            sink.write_rows([{"layer": "x", "tokens": 1, "time_us": 1.0}])
        assert p.exists()


def test_sink_unopened_write_raises():
    sink = CsvSink("/tmp/notopened.csv", DENSE_COLS)
    with pytest.raises(RuntimeError, match="未打开"):
        sink.write_rows([{"layer": "x", "tokens": 1, "time_us": 1.0}])


# ---- read_* ----

def test_read_dense_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dense.csv"
        with CsvSink(p, DENSE_COLS) as sink:
            sink.write_rows([
                {"layer": "qkv_proj", "tokens": 128, "time_us": 42.5},
                {"layer": "down_proj", "tokens": 256, "time_us": 87.1},
            ])
        rows = read_dense(p)
        assert rows == [
            DenseRow("qkv_proj", 128, 42.5),
            DenseRow("down_proj", 256, 87.1),
        ]


def test_read_attention_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "attn.csv"
        with CsvSink(p, ATTN_COLS) as sink:
            sink.write_rows([
                {"prefill_chunk": 512, "kv_prefill": 0,
                 "n_decode": 0, "kv_decode": 0, "time_us": 1234.0},
                {"prefill_chunk": 0, "kv_prefill": 0,
                 "n_decode": 4, "kv_decode": 2048, "time_us": 5.6},
            ])
        rows = read_attention(p)
        assert rows == [
            AttnRow(512, 0, 0, 0, 1234.0),
            AttnRow(0, 0, 4, 2048, 5.6),
        ]


def test_read_nonexistent_returns_empty():
    assert read_dense("/nonexistent/x.csv") == []
    assert read_attention("/nonexistent/x.csv") == []
    assert read_per_sequence("/nonexistent/x.csv") == []


# ---- samples → rows 转换 ----

def test_samples_to_dense_rows():
    shot = Shot(kind="dense", num_new_tokens=128)
    samples = [
        TimingSample(layer="qkv_proj", op_kind="dense_gemm", microseconds=42.0),
        TimingSample(layer="o_proj", op_kind="dense_gemm", microseconds=15.0),
    ]
    rows = samples_to_dense_rows(shot, samples)
    assert rows == [
        {"layer": "qkv_proj", "tokens": 128, "time_us": 42.0},
        {"layer": "o_proj", "tokens": 128, "time_us": 15.0},
    ]


def test_samples_to_per_seq_rows():
    shot = Shot(kind="per_sequence", num_new_tokens=4, num_decode_seqs=4,
                kv_lens_decode=[256] * 4)
    samples = [
        TimingSample(layer="lm_head", op_kind="dense_gemm", microseconds=100.0),
    ]
    rows = samples_to_per_seq_rows(shot, samples)
    assert rows == [{"layer": "lm_head", "sequences": 4, "time_us": 100.0}]


def test_samples_to_attn_rows_includes_4d_key():
    shot = Shot(kind="attention",
                num_new_tokens=128 + 4, prefill_chunk=128,
                num_decode_seqs=4,
                kv_lens_prefill=[0], kv_lens_decode=[2048] * 4)
    samples = [
        TimingSample(layer="attention", op_kind="attn", microseconds=500.0),
    ]
    rows = samples_to_attn_rows(shot, samples)
    assert rows == [{
        "prefill_chunk": 128, "kv_prefill": 0,
        "n_decode": 4, "kv_decode": 2048, "time_us": 500.0,
    }]


def test_samples_to_attn_rows_empty_kv_lens():
    """kv_lens_prefill/decode 为空时用 0."""
    shot = Shot(kind="attention", num_new_tokens=512, prefill_chunk=512)
    samples = [TimingSample(layer="attention", op_kind="attn", microseconds=100.0)]
    rows = samples_to_attn_rows(shot, samples)
    assert rows[0]["kv_prefill"] == 0
    assert rows[0]["kv_decode"] == 0


# ---- visited_keys (resume) ----

def test_visited_keys_dense():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dense.csv"
        with CsvSink(p, DENSE_COLS) as sink:
            sink.write_rows([
                {"layer": "qkv_proj", "tokens": 128, "time_us": 1.0},
                {"layer": "o_proj", "tokens": 128, "time_us": 2.0},
                {"layer": "qkv_proj", "tokens": 512, "time_us": 3.0},
            ])
        keys = visited_keys_dense(p)
        assert keys == {(128,), (512,)}


def test_visited_keys_attention():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "attn.csv"
        with CsvSink(p, ATTN_COLS) as sink:
            sink.write_rows([
                {"prefill_chunk": 512, "kv_prefill": 0,
                 "n_decode": 0, "kv_decode": 0, "time_us": 1.0},
                {"prefill_chunk": 0, "kv_prefill": 0,
                 "n_decode": 4, "kv_decode": 2048, "time_us": 2.0},
            ])
        keys = visited_keys_attention(p)
        assert keys == {(512, 0, 0, 0), (0, 0, 4, 2048)}


def test_visited_keys_per_sequence():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ps.csv"
        with CsvSink(p, PER_SEQ_COLS) as sink:
            sink.write_rows([
                {"layer": "lm_head", "sequences": 4, "time_us": 1.0},
                {"layer": "lm_head", "sequences": 16, "time_us": 2.0},
            ])
        keys = visited_keys_per_sequence(p)
        assert keys == {(4,), (16,)}


def test_visited_keys_missing_file():
    assert visited_keys_dense("/nonexistent/x.csv") == set()
