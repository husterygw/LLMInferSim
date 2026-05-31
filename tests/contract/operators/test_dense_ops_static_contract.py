"""Phase 3 static + forward contract equivalence for roofline-only dense ops
(Norm / ElementWise / Embedding). Same safety property as GEMM (Phase 2): the
static op (tokens from StepRuntime via forward()) must be numerically identical
to the legacy op (tokens baked) so the engine switch moves no sim number."""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.step.runtime import StepRuntime
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.operators.elementwise import ElementWise
from llm_infer_sim.core.operators.embedding import Embedding
from llm_infer_sim.core.operators.norm import Norm
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from tests.helpers.support import make_model_config

_T = 2048


def _ctx():
    return build_operator_context(
        make_model_config(hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
                    ffn_dim=6144, num_layers=48, vocab_size=151936),
        DeploymentProfile.flat(tp=1),
        RuntimeProfile.flat(execution_mode="cudagraph", backend="vllm",
                            backend_version="0.19.1"),
        get_hardware_profile("RTX_4090"),
    )


def _step(t=_T):
    return StepRuntime(phase="prefill", total_tokens=t, num_prefill_tokens=t,
                       num_prefill_requests=1, execution_mode="cudagraph")


def _pairs(ctx):
    """(legacy_op, static_op) pairs that should be equivalent at tokens=_T."""
    tok = lambda s: s.total_tokens  # noqa: E731
    return [
        (Norm(name="n", op_subtype="attn_norm", phase="prefill", layer_idx=0,
              tokens=_T, hidden=2048, ctx=ctx),
         Norm(name="n", op_subtype="attn_norm", phase="prefill", layer_idx=None,
              tokens=0, hidden=2048, ctx=ctx, count=48, tokens_fn=tok)),
        (ElementWise(name="e", op_subtype="mlp_act", phase="prefill", layer_idx=0,
                     tokens=_T, intermediate=768, ctx=ctx),
         ElementWise(name="e", op_subtype="mlp_act", phase="prefill", layer_idx=None,
                     tokens=0, intermediate=768, ctx=ctx, count=48, tokens_fn=tok)),
        (ElementWise(name="r", op_subtype="rope", phase="prefill", layer_idx=0,
                     tokens=_T, num_heads=40, head_dim=128, ctx=ctx),
         ElementWise(name="r", op_subtype="rope", phase="prefill", layer_idx=None,
                     tokens=0, num_heads=40, head_dim=128, ctx=ctx, count=48, tokens_fn=tok)),
        (Embedding(name="emb", phase="prefill", layer_idx=0, tokens=_T,
                   vocab_size=151936, hidden=2048, ctx=ctx),
         Embedding(name="emb", phase="prefill", layer_idx=None, tokens=0,
                   vocab_size=151936, hidden=2048, ctx=ctx, tokens_fn=tok)),
    ]


def test_roofline_spec_equivalent_to_legacy():
    ctx = _ctx()
    for legacy, static in _pairs(ctx):
        assert static.roofline_spec(static.forward(_step())) == legacy.roofline_spec()


def test_backend_latency_equivalent():
    ctx = _ctx()
    rl = RooflineBackend(ctx.hw, ctx.execution_mode)
    for legacy, static in _pairs(ctx):
        legacy_e = rl.estimate(legacy)
        new_e = rl.estimate(static, static.forward(_step()))
        assert new_e.latency_s == legacy_e.latency_s


def test_forward_uses_step_tokens():
    ctx = _ctx()
    for _, static in _pairs(ctx):
        rt = static.forward(_step(t=512))
        assert rt.shape["tokens"] == 512
        # different token counts give different roofline (sanity)
        assert static.roofline_spec(static.forward(_step(t=512))) != \
            static.roofline_spec(static.forward(_step(t=2048)))
