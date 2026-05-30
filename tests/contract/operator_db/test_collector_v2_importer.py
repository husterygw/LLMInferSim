"""collector_v2 importer 单测 — Step 3.4.

锁住:
  - GEMM RawRecord -> OperatorRecord 字段映射
  - signature 与同 shape collector case canonicalizer 一致
  - 真 JSONL file 可加载 (RTX_4090 / vllm-0.19.1 / gemm.jsonl)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from llm_infer_sim.core.operator_db.importers.collector_v2 import (
    import_record,
    raw_record_to_signature,
)
from llm_infer_sim.core.operator_db.stores.jsonl import JsonlOperatorStore
from llm_infer_sim.core.operator_schema.gemm import gemm_case_params_to_signature


_GEMM_ROW = {
    "case_id": "gemm__qkv_proj_tp1_m1_eager__abc",
    "op_kind": "gemm",
    "framework": "vllm",
    "framework_version": "0.19.1",
    "device": "NVIDIA GeForce RTX 4090",
    "execution_mode": "eager",
    "kernel_source": "vllm_row_parallel_linear",
    "params": {
        "op_subtype": "qkv_proj", "m": 1, "n": 6144, "k": 2560,
        "dtype": "bf16", "tp": 1, "execution_mode": "eager",
    },
    "metrics": {
        "latency_us_p50": 46.6, "latency_us_p10": 46.0, "latency_us_p90": 63.0,
        "used_cuda_graph": False, "n_warmups": 3, "n_iters": 10,
    },
    "metadata": {"fallback_reason": None, "source_profiles": ["qwen3_4b"]},
    "schema_version": "collector-v2",
}


def test_import_record_basic_fields():
    rec = import_record(_GEMM_ROW, hardware="RTX_4090")
    assert rec.hardware == "RTX_4090"
    assert rec.framework == "vllm"
    assert rec.framework_version == "0.19.1"
    assert rec.execution_mode == "eager"
    assert rec.kernel_source == "vllm_row_parallel_linear"
    assert rec.latency_us_p50 == 46.6
    assert rec.n_iters == 10
    assert rec.n_warmups == 3
    assert rec.source["case_id"] == "gemm__qkv_proj_tp1_m1_eager__abc"
    assert rec.source["source_profiles"] == ["qwen3_4b"]


def test_signature_matches_stage_2_gemm_canonicalizer():
    """importer 生成的 signature 必须与直接用 gemm_case_params_to_signature 等价."""
    sig_via_importer = raw_record_to_signature(_GEMM_ROW)
    sig_direct = gemm_case_params_to_signature(
        _GEMM_ROW["params"],
        framework="vllm",
        framework_version="0.19.1",
        kernel_source="vllm_row_parallel_linear",
    )
    assert sig_via_importer == sig_direct
    assert sig_via_importer.stable_hash() == sig_direct.stable_hash()


def test_attention_record_imports_with_inferred_backend():
    row = {
        "case_id": "attn_x", "op_kind": "attention",
        "framework": "vllm", "framework_version": "0.19.1",
        "device": "RTX_4090", "execution_mode": "eager",
        "kernel_source": "vllm_flash_attn",
        "params": {
            "phase": "prefill", "batch_size": 1, "isl": 128,
            "kv_prefill": 0, "n_decode": 0, "kv_decode": 0,
            "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
            "dtype": "bf16", "tp": 1, "execution_mode": "eager",
        },
        "metrics": {
            "latency_us_p50": 100.0, "latency_us_p10": 95, "latency_us_p90": 110,
            "n_iters": 10, "n_warmups": 3,
        },
        "metadata": {},
    }
    rec = import_record(row, hardware="RTX_4090")
    runtime = dict(rec.signature.runtime)
    assert runtime["attention_backend"] == "flash_attn"
    assert runtime["kv_dtype"] == "bf16"
    assert runtime["block_size"] == 16


def test_unknown_op_kind_raises():
    bad = dict(_GEMM_ROW)
    bad["op_kind"] = "unknown_kind"
    with pytest.raises(ValueError, match="unknown op_kind"):
        raw_record_to_signature(bad)


# ---- 真 JSONL 加载 ----

_REAL_PATH = Path(
    "collector/data/operator_db/RTX_4090/vllm-0.19.1/gemm.jsonl"
)


@pytest.mark.skipif(not _REAL_PATH.exists(), reason="real JSONL not present")
def test_jsonl_store_loads_real_gemm_dataset():
    store = JsonlOperatorStore.from_jsonl(_REAL_PATH, hardware="RTX_4090")
    # 实际数据 144 行
    assert len(store) >= 100, f"expected ≥100 records, got {len(store)}"


@pytest.mark.skipif(not _REAL_PATH.exists(), reason="real JSONL not present")
def test_partition_loader_skips_missing_op_kinds(tmp_path):
    # 把真 JSONL 复制到 partition layout 让 load_partition 找到
    fake_root = tmp_path / "operator_db"
    partition = fake_root / "RTX_4090" / "vllm-0.19.1"
    partition.mkdir(parents=True)
    (partition / "gemm.jsonl").write_text(_REAL_PATH.read_text())
    store = JsonlOperatorStore()
    counts = store.load_partition(
        fake_root, hardware="RTX_4090",
        framework="vllm", framework_version="0.19.1",
        # this test exercises op_kind skipping, not the runtime-version precheck;
        # opt out so it doesn't fail-closed on the 0.19.1-vs-runtime skew.
        verify_version=False,
    )
    assert counts["gemm"] > 0
    # attention/moe/collective 不在 partition 里 → 0, 不抛错
    assert counts["attention"] == 0
    assert counts["moe"] == 0
    assert counts["collective"] == 0
