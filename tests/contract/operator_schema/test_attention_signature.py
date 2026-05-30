"""Attention canonicalizer 单测 — Step 2.5."""
from __future__ import annotations

import pytest

from llm_infer_sim.core.operator_schema.attention import (
    attention_case_params_to_signature,
    attention_operator_to_signature,
)
from types import SimpleNamespace

from llm_infer_sim.core.operators import Attention
from llm_infer_sim.core.operators.base import RooflineSpec


_RUNTIME_CTX = dict(
    framework="vllm",
    framework_version="0.20.1",
    kernel_source="vllm_flash_attn",
    attention_backend="flash_attn",
    kv_dtype="bf16",
    block_size=16,
)


def _prefill_case(isl=2048, bs=1, tp=1, mode="eager"):
    return {
        "phase": "prefill", "batch_size": bs, "isl": isl,
        "kv_prefill": 0, "n_decode": 0, "kv_decode": 0,
        "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "dtype": "bf16", "tp": tp, "execution_mode": mode,
    }


def _decode_case(n=8, ctx=2048, tp=1, mode="eager"):
    return {
        "phase": "decode", "batch_size": 0, "isl": 0,
        "kv_prefill": 0, "n_decode": n, "kv_decode": ctx,
        "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "dtype": "bf16", "tp": tp, "execution_mode": mode,
    }


def _attn_ctx(tp=1, mode="eager"):
    from llm_infer_sim.core.operators.context import build_operator_context
    from llm_infer_sim.core.deployment.profile import DeploymentProfile
    from llm_infer_sim.core.runtime.profile import RuntimeProfile
    from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
    from tests.helpers.support import make_model_config
    return build_operator_context(
        make_model_config(),
        DeploymentProfile.flat(tp=tp, block_size=16),
        RuntimeProfile.flat(
            execution_mode=mode, backend="vllm", backend_version="0.20.1",
        ),
        get_hardware_profile("RTX_4090"),
    )


def _prefill_op(isl=2048, bs=1, tp=1, mode="eager"):
    return Attention(
        name="attention", op_subtype="prefill",
        phase="prefill", layer_idx=0,
        num_tokens=bs * isl, num_seqs=bs,
        q_len=isl, kv_len=isl,
        num_q_heads=32, num_kv_heads=8, head_dim=128,
        attention_backend="flash_attn", kv_dtype="bf16", block_size=16,
        kernel_source="vllm_flash_attn",
        ctx=_attn_ctx(tp=tp, mode=mode),
        roofline_spec_value=RooflineSpec(flops=1, op_category="attention"),
    )


def _decode_op(n=8, ctx=2048, tp=1, mode="eager"):
    return Attention(
        name="attention", op_subtype="decode",
        phase="decode", layer_idx=0,
        num_tokens=n, num_seqs=n,
        q_len=1, kv_len=ctx,
        num_q_heads=32, num_kv_heads=8, head_dim=128,
        attention_backend="flash_attn", kv_dtype="bf16", block_size=16,
        kernel_source="vllm_flash_attn",
        ctx=_attn_ctx(tp=tp, mode=mode),
        roofline_spec_value=RooflineSpec(flops=1, op_category="attention"),
    )


def test_prefill_collector_and_runtime_signature_match():
    sig_c = attention_case_params_to_signature(_prefill_case(), **_RUNTIME_CTX)
    sig_r = attention_operator_to_signature(_prefill_op())
    assert sig_c == sig_r
    assert sig_c.stable_hash() == sig_r.stable_hash()


def test_decode_collector_and_runtime_signature_match():
    sig_c = attention_case_params_to_signature(_decode_case(), **_RUNTIME_CTX)
    sig_r = attention_operator_to_signature(_decode_op())
    assert sig_c == sig_r
    assert sig_c.stable_hash() == sig_r.stable_hash()


def test_prefill_vs_decode_signatures_differ():
    sig_p = attention_case_params_to_signature(_prefill_case(), **_RUNTIME_CTX)
    sig_d = attention_case_params_to_signature(_decode_case(), **_RUNTIME_CTX)
    assert sig_p != sig_d
    assert sig_p.op_subtype == "prefill"
    assert sig_d.op_subtype == "decode"


def test_prefill_q_len_equals_kv_len_equals_isl():
    sig = attention_case_params_to_signature(_prefill_case(isl=2048), **_RUNTIME_CTX)
    shape = dict(sig.shape)
    assert shape["q_len"] == 2048
    assert shape["kv_len"] == 2048
    assert shape["num_tokens"] == 2048   # bs=1


def test_decode_q_len_one_kv_len_ctx():
    sig = attention_case_params_to_signature(_decode_case(n=8, ctx=1024), **_RUNTIME_CTX)
    shape = dict(sig.shape)
    assert shape["q_len"] == 1
    assert shape["kv_len"] == 1024
    assert shape["num_seqs"] == 8
    assert shape["num_tokens"] == 8


def test_eager_and_cudagraph_differ():
    sig_e = attention_case_params_to_signature(_prefill_case(mode="eager"), **_RUNTIME_CTX)
    sig_g = attention_case_params_to_signature(_prefill_case(mode="cudagraph"), **_RUNTIME_CTX)
    assert sig_e != sig_g


def test_attention_backend_in_runtime_key():
    """flash_attn vs flashinfer 应该是不同 signature."""
    ctx_flash = dict(_RUNTIME_CTX)
    ctx_flash["attention_backend"] = "flash_attn"
    ctx_flashinfer = dict(_RUNTIME_CTX)
    ctx_flashinfer["attention_backend"] = "flashinfer"
    sig_a = attention_case_params_to_signature(_prefill_case(), **ctx_flash)
    sig_b = attention_case_params_to_signature(_prefill_case(), **ctx_flashinfer)
    assert sig_a != sig_b


def test_kv_dtype_in_runtime_key():
    ctx_bf16 = dict(_RUNTIME_CTX); ctx_bf16["kv_dtype"] = "bf16"
    ctx_fp8 = dict(_RUNTIME_CTX); ctx_fp8["kv_dtype"] = "fp8"
    sig_a = attention_case_params_to_signature(_prefill_case(), **ctx_bf16)
    sig_b = attention_case_params_to_signature(_prefill_case(), **ctx_fp8)
    assert sig_a != sig_b


def test_block_size_in_runtime_key():
    ctx_16 = dict(_RUNTIME_CTX); ctx_16["block_size"] = 16
    ctx_32 = dict(_RUNTIME_CTX); ctx_32["block_size"] = 32
    sig_a = attention_case_params_to_signature(_prefill_case(), **ctx_16)
    sig_b = attention_case_params_to_signature(_prefill_case(), **ctx_32)
    assert sig_a != sig_b


def test_mixed_phase_not_implemented():
    bad = _prefill_case()
    bad["phase"] = "mixed"
    with pytest.raises(NotImplementedError, match="prefill / decode"):
        attention_case_params_to_signature(bad, **_RUNTIME_CTX)


def test_operator_wrong_kind_raises():
    bogus = SimpleNamespace(op_kind="gemm")
    with pytest.raises(ValueError, match="op_kind=attention"):
        attention_operator_to_signature(bogus)
