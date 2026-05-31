"""Phase 3 Collective (AllReduce) static + forward contract equivalence.

Same safety property as GEMM: static AllReduce (message_bytes from StepRuntime
via forward()) must be byte-for-byte identical to the legacy op — same signature
hash, same backend (communication-model) latency — so the engine switch moves no
sim number. Locks the口径 fix too: signature carries kernel_source + topology.
"""
from __future__ import annotations

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.step.runtime import StepRuntime
from llm_infer_sim.core.operators.collective import AllReduce
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from tests.helpers.support import make_model_config

_MB = 2048 * 2048 * 2  # tokens * hidden * 2 bytes


def _ctx(tp=4):
    return build_operator_context(
        make_model_config(hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
                    ffn_dim=6144, num_layers=48, vocab_size=151936),
        DeploymentProfile.flat(tp=tp),
        RuntimeProfile.flat(execution_mode="cudagraph", backend="vllm",
                            backend_version="0.19.1"),
        get_hardware_profile("RTX_4090"),
    )


def _legacy(ctx):
    return AllReduce(name="tp_o_proj_allreduce", phase="prefill", layer_idx=0,
                     message_bytes=_MB, world_size=4, ctx=ctx,
                     kernel_source="torch_dist_nccl", topology="concentrated")


def _static(ctx):
    return AllReduce(name="tp_o_proj_allreduce", phase="prefill", layer_idx=None,
                     message_bytes=0, world_size=4, ctx=ctx,
                     kernel_source="torch_dist_nccl", topology="concentrated",
                     count=48, message_bytes_fn=lambda s: s.total_tokens * 2048 * 2)


def _step(t=2048):
    return StepRuntime(phase="prefill", total_tokens=t, num_prefill_tokens=t,
                       num_prefill_requests=1, execution_mode="cudagraph")


def test_forward_computes_message_bytes_from_step():
    rt = _static(_ctx()).forward(_step(t=512))
    assert rt.shape["message_bytes"] == 512 * 2048 * 2


def test_signature_equivalent_to_legacy():
    ctx = _ctx()
    new = _static(ctx).signature(_static(ctx).forward(_step()))
    legacy = _legacy(ctx).signature()
    assert new == legacy and new.stable_hash() == legacy.stable_hash()
    #口径 fix locked: topology + canonical kernel_source in the signature
    assert dict(new.runtime)["topology"] == "concentrated"
    assert dict(new.runtime)["kernel_source"] == "torch_dist_nccl"


def test_backend_latency_equivalent():
    ctx = _ctx()
    rl = RooflineBackend(ctx.hw, ctx.execution_mode)
    legacy_e = rl.estimate(_legacy(ctx))
    new_e = rl.estimate(_static(ctx), _static(ctx).forward(_step()))
    assert new_e.latency_s == legacy_e.latency_s
    assert new_e.metadata.get("bottleneck") == "communication"
