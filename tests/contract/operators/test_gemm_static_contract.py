"""Phase 2 GEMM static + forward contract equivalence (op_plan §6).

The safety property for the eventual engine switch: a STATIC GEMM (m computed
from StepRuntime via forward()) must be byte-for-byte identical to the LEGACY
GEMM (m baked at construction) — same signature hash, same roofline spec, same
backend latency, same DB hit. If this holds, switching the engine to forward()
moves no sim number.
"""
from __future__ import annotations

from llm_infer_sim.core.cost.backends.operator_db import OperatorDBBackend
from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.graph.runtime import StepRuntime
from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_db.stores.memory import MemoryOperatorStore
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.operators.gemm import GEMM
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from tests.helpers.support import make_model_config

_M, _N, _K = 2048, 5120, 2048


def _ctx(tp=1):
    return build_operator_context(
        make_model_config(hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
                    ffn_dim=6144, num_layers=48, vocab_size=151936),
        DeploymentProfile.flat(tp=tp),
        RuntimeProfile.flat(execution_mode="cudagraph",
                            backend="vllm", backend_version="0.19.1"),
        get_hardware_profile("RTX_4090"),
    )


def _legacy(ctx):
    return GEMM(name="qkv_proj", op_subtype="qkv_proj", phase="prefill",
               layer_idx=0, m=_M, n=_N, k=_K, ctx=ctx,
               kernel_source="vllm_row_parallel_linear")


def _static(ctx, count=36):
    # m not baked: computed from StepRuntime by forward(). m=0 placeholder.
    return GEMM(name="qkv_proj", op_subtype="qkv_proj", phase="prefill",
                layer_idx=None, m=0, n=_N, k=_K, ctx=ctx,
                kernel_source="vllm_row_parallel_linear",
                count=count, m_fn=lambda s: s.total_tokens)


def _step(total_tokens=_M):
    return StepRuntime(phase="prefill", total_tokens=total_tokens,
                       num_prefill_tokens=total_tokens, num_prefill_requests=1,
                       execution_mode="cudagraph")


def test_forward_computes_m_from_step():
    ctx = _ctx()
    rt = _static(ctx).forward(_step(total_tokens=512))
    assert rt.shape["m"] == 512 and rt.shape["n"] == _N and rt.shape["k"] == _K


def test_signature_equivalent_to_legacy():
    ctx = _ctx()
    legacy_sig = _legacy(ctx).signature()
    rt = _static(ctx).forward(_step())
    new_sig = _static(ctx).signature(rt)
    assert new_sig == legacy_sig
    assert new_sig.stable_hash() == legacy_sig.stable_hash()


def test_roofline_spec_equivalent_to_legacy():
    ctx = _ctx()
    assert _static(ctx).roofline_spec(_static(ctx).forward(_step())) == _legacy(ctx).roofline_spec()


def test_backend_roofline_latency_equivalent():
    ctx = _ctx()
    rl = RooflineBackend(ctx.hw, ctx.execution_mode)
    legacy_entry = rl.estimate(_legacy(ctx))
    new_entry = rl.estimate(_static(ctx), _static(ctx).forward(_step()))
    assert new_entry.latency_s == legacy_entry.latency_s
    assert new_entry.metadata["flops"] == legacy_entry.metadata["flops"]


def test_db_hit_equivalent():
    ctx = _ctx()
    legacy = _legacy(ctx)
    store = MemoryOperatorStore()
    store.add(OperatorRecord(
        signature=legacy.signature(), framework="vllm", framework_version="0.19.1",
        hardware="RTX_4090", execution_mode="cudagraph", kernel_source="vllm_row_parallel_linear",
        latency_us_p50=123.0, latency_us_p10=120.0, latency_us_p90=126.0,
        n_iters=10, n_warmups=3, source={"case_id": "x"},
    ))
    db = OperatorDBBackend(store, roofline=RooflineBackend(ctx.hw, ctx.execution_mode))
    # legacy path hits
    legacy_hit = db.estimate(legacy)
    assert legacy_hit is not None
    # static-contract path (forward → op_runtime) hits the SAME record
    rt = _static(ctx).forward(_step())
    hit = db.estimate(_static(ctx), rt)
    assert hit is not None
    assert hit.source == "operator_db"
    assert hit.latency_s == legacy_hit.latency_s


def test_different_step_tokens_change_signature():
    ctx = _ctx()
    op = _static(ctx)
    sig_a = op.signature(op.forward(_step(total_tokens=512)))
    sig_b = op.signature(op.forward(_step(total_tokens=2048)))
    assert sig_a != sig_b  # M is part of the GEMM signature
