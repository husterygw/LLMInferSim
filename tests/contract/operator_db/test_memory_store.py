"""MemoryOperatorStore 单测 — Step 3.3."""
from __future__ import annotations

from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_db.stores.memory import MemoryOperatorStore
from llm_infer_sim.core.operator_schema.signature import OperatorSignature


def _make_record(m=128, latency_us=100.0) -> OperatorRecord:
    sig = OperatorSignature(
        op_kind="gemm", op_subtype="qkv_proj", dtype="bf16",
        shape=(("k", 2560), ("m", m), ("n", 6144)),
        parallel=(("tp", 1),),
        runtime=(
            ("execution_mode", "eager"),
            ("framework", "vllm"),
            ("framework_version", "0.19.1"),
            ("kernel_source", "vllm_row_parallel_linear"),
        ),
    )
    return OperatorRecord(
        signature=sig, hardware="RTX_4090",
        framework="vllm", framework_version="0.19.1",
        execution_mode="eager", kernel_source="vllm_row_parallel_linear",
        latency_us_p50=latency_us, latency_us_p10=latency_us * 0.9,
        latency_us_p90=latency_us * 1.1,
        n_iters=10, n_warmups=3,
    )


def test_add_and_lookup_hit():
    store = MemoryOperatorStore()
    rec = _make_record()
    store.add(rec)
    assert len(store) == 1
    hit = store.lookup(rec.signature)
    assert hit is rec


def test_lookup_miss_returns_none():
    store = MemoryOperatorStore()
    rec = _make_record(m=128)
    store.add(rec)
    other = _make_record(m=2048)
    assert store.lookup(other.signature) is None


def test_add_overwrites_same_signature():
    store = MemoryOperatorStore()
    a = _make_record(latency_us=100.0)
    store.add(a)
    b = OperatorRecord(
        signature=a.signature,    # same sig
        hardware="RTX_4090",
        framework="vllm", framework_version="0.19.1",
        execution_mode="eager", kernel_source="vllm_row_parallel_linear",
        latency_us_p50=200.0, latency_us_p10=180.0, latency_us_p90=220.0,
        n_iters=10, n_warmups=3,
    )
    store.add(b)
    assert len(store) == 1
    hit = store.lookup(a.signature)
    assert hit is b
    assert hit.latency_us_p50 == 200.0


def test_iter_yields_all_records():
    store = MemoryOperatorStore()
    a = _make_record(m=128)
    b = _make_record(m=2048)
    store.add(a)
    store.add(b)
    assert len(store) == 2
    seen = set(r.signature.stable_hash() for r in store)
    assert seen == {a.signature.stable_hash(), b.signature.stable_hash()}


def test_record_latency_s_property():
    import pytest
    rec = _make_record(latency_us=123.0)
    assert rec.latency_s == pytest.approx(123e-6)
