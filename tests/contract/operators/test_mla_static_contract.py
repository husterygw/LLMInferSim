"""Phase 5 (③) MLA static + forward recompute equivalence (DeepSeek path).

The highest-risk migration — DeepSeek has no bench oracle. These lock that the
MLA spec recomputed from a step-resolved shape (forward → spec_fn) is byte-for-
byte identical to the constructor's baked spec, for dense MLA, prefill + decode.
(Real prefill has ctx_len == max_prefill_seqlen, i.e. context >= current tokens,
which is what forward derives.)"""
from __future__ import annotations

from llm_infer_sim.core.step.runtime import StepRuntime
from llm_infer_sim.core.operators.mla import MLAAttention
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from tests.helpers.support import make_model_config

_SEQ, _BS_P, _CTXLEN, _BS_D = 2048, 1, 4096, 8
# DeepSeek-V3 MLA dims
_HEADS, _QK, _V, _KVLAT, _KVLORA = 16, 192, 128, 576, 512


def _ctx(tp=8):
    return build_operator_context(
        make_model_config(hidden_dim=7168, num_heads=128, num_kv_heads=128, head_dim=56,
                    ffn_dim=18432, num_layers=61, vocab_size=129280,
                    kv_lora_rank=512, kv_latent_dim=576),
        DeploymentProfile.flat(tp=tp), RuntimeProfile.flat(),
        get_hardware_profile("RTX_4090"),
    )


def _prefill_step():
    # context == seqlen (fresh prefill): max_context_len == max_prefill_seqlen
    return StepRuntime(phase="prefill", total_tokens=_SEQ * _BS_P,
                       num_prefill_tokens=_SEQ * _BS_P, num_prefill_requests=_BS_P,
                       max_prefill_seqlen=_SEQ, max_context_len=_SEQ,
                       execution_mode="cudagraph")


def _decode_step():
    return StepRuntime(phase="decode", total_tokens=_BS_D, num_decode_tokens=_BS_D,
                       num_decode_requests=_BS_D, avg_decode_context_len=_CTXLEN,
                       execution_mode="cudagraph")


def _ops(ctx):
    """(label, op, step) for each MLA variant; ctx_len=_SEQ for prefill (fresh)."""
    return [
        ("mla_prefill", MLAAttention.mla_prefill(
            layer_idx=0, seqlen=_SEQ, bs=_BS_P, ctx_len=_SEQ, heads_per_tp=_HEADS,
            qk_head_dim=_QK, v_dim=_V, kv_latent_dim=_KVLAT, ctx=ctx), _prefill_step()),
        ("mla_decode", MLAAttention.mla_decode(
            layer_idx=0, ctx_len=_CTXLEN, bs=_BS_D, heads_per_tp=_HEADS,
            kv_latent_dim=_KVLAT, kv_lora_rank=_KVLORA, ctx=ctx), _decode_step()),
    ]


def test_mla_recompute_equals_baked():
    ctx = _ctx()
    for label, op, step in _ops(ctx):
        rt = op.forward(step)
        assert rt is not None, label
        assert op.roofline_spec(rt) == op.roofline_spec(), label  # recompute == baked
        assert op.signature(rt) == op.signature(), label
        assert op.signature(rt).stable_hash() == op.signature().stable_hash(), label


def test_mla_forward_shape_matches_baked():
    ctx = _ctx()
    for label, op, step in _ops(ctx):
        rt = op.forward(step)
        # forward-derived shape must equal the constructor's baked shape
        assert dict(rt.shape) == op.shape, label
