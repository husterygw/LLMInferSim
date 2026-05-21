"""OperatorSignature 单测 — Step 2.1."""
from __future__ import annotations

import dataclasses

import pytest

from llm_infer_sim.core.operator_schema import (
    OperatorSignature,
    project,
    to_canonical,
)


def _sample() -> OperatorSignature:
    return OperatorSignature(
        op_kind="gemm",
        op_subtype="qkv_proj",
        dtype="bf16",
        shape=(("k", 2560), ("m", 128), ("n", 6144)),
        parallel=(("tp", 1),),
        runtime=(
            ("execution_mode", "eager"),
            ("framework", "vllm"),
            ("framework_version", "0.20.1"),
            ("kernel_source", "vllm_default"),
        ),
    )


def test_frozen():
    sig = _sample()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sig.op_kind = "other"  # type: ignore[misc]


def test_hashable_and_equal_for_same_fields():
    a = _sample()
    b = _sample()
    assert a == b
    assert hash(a) == hash(b)


def test_stable_hash_is_hex_digest():
    sig = _sample()
    h = sig.stable_hash()
    assert isinstance(h, str)
    assert len(h) == 64
    assert int(h, 16) >= 0


def test_stable_hash_changes_when_field_changes():
    a = _sample()
    b = OperatorSignature(
        op_kind="gemm", op_subtype="qkv_proj", dtype="bf16",
        shape=(("k", 2560), ("m", 2048), ("n", 6144)),  # m 不同
        parallel=(("tp", 1),),
        runtime=a.runtime,
    )
    assert a.stable_hash() != b.stable_hash()


def test_to_json_dict_is_serializable():
    import json
    sig = _sample()
    d = sig.to_json_dict()
    s = json.dumps(d)
    assert json.loads(s) == d


# ---- canonical helpers ----

def test_to_canonical_sorts_alphabetically():
    raw = {"n": 6144, "m": 128, "k": 2560}
    assert to_canonical(raw) == (("k", 2560), ("m", 128), ("n", 6144))


def test_to_canonical_drops_none():
    raw = {"a": 1, "b": None, "c": 3}
    assert to_canonical(raw) == (("a", 1), ("c", 3))


def test_project_missing_yields_none():
    """project 缺失字段 → None (供 to_canonical 跳过)."""
    raw = {"a": 1}
    p = project(raw, ("a", "b"))
    assert p == {"a": 1, "b": None}
    assert to_canonical(p) == (("a", 1),)
