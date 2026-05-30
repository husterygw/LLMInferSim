"""Phase 1 signature-field registry + validating builder (op_plan §2).

Locks the contract that turned the silent DB-miss into a loud failure:
unregistered signature fields raise instead of dropping out via project().
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.operator_schema.fields import (
    SIGNATURE_FIELDS,
    SignatureFieldError,
    build_validated_signature,
)
from llm_infer_sim.core.operator_schema.gemm import gemm_case_params_to_signature


def test_registry_sourced_from_canonicalizers():
    assert SIGNATURE_FIELDS["gemm"]["shape"] == frozenset({"m", "n", "k"})
    assert SIGNATURE_FIELDS["moe"]["parallel"] == frozenset({"tp", "ep"})
    # collective topology must be a registered runtime field (口径 fix depends on it)
    assert "topology" in SIGNATURE_FIELDS["collective"]["runtime"]
    assert "message_bytes" in SIGNATURE_FIELDS["collective"]["shape"]


def test_built_signature_matches_existing_canonicalizer():
    """build_validated_signature must be byte-identical to the legacy path so
    migrating canonicalizers onto it doesn't move any signature/hash."""
    built = build_validated_signature(
        op_kind="gemm", op_subtype="qkv_proj", dtype="bf16",
        shape={"m": 2048, "n": 1280, "k": 2048}, parallel={"tp": 4},
        runtime={"framework": "vllm", "framework_version": "0.19.1",
                 "execution_mode": "cudagraph", "kernel_source": "vllm_row_parallel_linear"},
    )
    ref = gemm_case_params_to_signature(
        {"m": 2048, "n": 1280, "k": 2048, "tp": 4, "execution_mode": "cudagraph",
         "op_subtype": "qkv_proj", "dtype": "bf16"},
        framework="vllm", framework_version="0.19.1",
        kernel_source="vllm_row_parallel_linear",
    )
    assert built == ref
    assert built.stable_hash() == ref.stable_hash()


@pytest.mark.parametrize("part,bad", [
    ("shape", {"msg_bytes": 131072}),          # collective message_bytes typo
    ("parallel", {"world_size": 4, "tpx": 1}),
    ("runtime", {"framewrok": "vllm"}),
])
def test_unregistered_field_raises(part, bad):
    kwargs = dict(op_kind="collective", op_subtype="allreduce", dtype="bf16",
                  shape={"message_bytes": 1}, parallel={"world_size": 2}, runtime={})
    kwargs[part] = bad
    with pytest.raises(SignatureFieldError):
        build_validated_signature(**kwargs)


def test_unknown_op_kind_raises():
    with pytest.raises(SignatureFieldError):
        build_validated_signature(op_kind="norm", op_subtype="x", dtype="bf16",
                                  shape={}, parallel={}, runtime={})


def test_none_valued_fields_are_ignored():
    """None means 'not applicable' (canonicalizer convention) — must not trip
    the unregistered-field check."""
    sig = build_validated_signature(
        op_kind="collective", op_subtype="allreduce", dtype="bf16",
        shape={"message_bytes": 131072},
        parallel={"world_size": 4, "tp": None, "ep": None},
        runtime={"framework": "vllm", "framework_version": "0.19.1",
                 "topology": "concentrated", "execution_mode": "cudagraph",
                 "kernel_source": "torch_dist_nccl", "algo": None, "protocol": None},
    )
    assert dict(sig.parallel)["world_size"] == 4
    assert dict(sig.runtime)["topology"] == "concentrated"
