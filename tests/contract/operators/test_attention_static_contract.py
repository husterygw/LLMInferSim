"""Attention single-op + forward contract (flash/GQA regime).

一个静态 Attention.flash() op, forward(step) 解析 prefill / decode / mixed:
  - prefill / decode: roofline_spec 从 step shape 重算, 与直接 _flash_*_spec 公式一致。
  - mixed: 单 op_subtype="mixed", roofline_spec = prefill 段 + decode 段 合成
    (_add_specs)。
MLA 在 test_mla_static_contract (operators/mla.py)。"""
from __future__ import annotations

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.step.runtime import StepRuntime
from llm_infer_sim.core.operators.attention import (
    Attention, _add_specs, _flash_decode_spec, _flash_prefill_spec,
)
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from tests.helpers.support import make_model_config

_SEQ, _BS_P = 2048, 1
_CTXLEN, _BS_D = 512, 8


def _ctx(tp=4):
    return build_operator_context(
        make_model_config(hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
                    ffn_dim=6144, num_layers=48, vocab_size=151936),
        DeploymentProfile.flat(tp=tp),
        RuntimeProfile.flat(execution_mode="cudagraph", backend="vllm",
                            backend_version="0.19.1"),
        get_hardware_profile("RTX_4090"),
    )


def _op(ctx):
    m = ctx.model
    return Attention.flash(layer_idx=0, n_q=m.num_heads // ctx.tp_size,
                           n_kv=m.num_kv_heads // ctx.tp_size, head_dim=m.head_dim,
                           ctx=ctx)


def _bytes(ctx):
    return dict(a_byte=ctx.a_byte, kv_byte=ctx.kv_byte, onchip=ctx.hw.onchip_buffer)


def _prefill_step(phase="prefill"):
    return StepRuntime(phase=phase, total_tokens=_SEQ * _BS_P,
                       num_prefill_tokens=_SEQ * _BS_P, num_prefill_requests=_BS_P,
                       max_prefill_seqlen=_SEQ, execution_mode="cudagraph")


def _decode_step(phase="decode"):
    return StepRuntime(phase=phase, total_tokens=_BS_D, num_decode_tokens=_BS_D,
                       num_decode_requests=_BS_D, avg_decode_context_len=_CTXLEN,
                       execution_mode="cudagraph")


def _mixed_step():
    return StepRuntime(phase="mixed", total_tokens=_SEQ + _BS_D,
                       num_prefill_tokens=_SEQ, num_prefill_requests=1,
                       num_decode_tokens=_BS_D, num_decode_requests=_BS_D,
                       max_prefill_seqlen=_SEQ, avg_decode_context_len=_CTXLEN,
                       execution_mode="cudagraph")


def test_prefill_resolves_and_spec_matches_formula():
    ctx = _ctx()
    nq, nkv, hd = ctx.model.num_heads // ctx.tp_size, ctx.model.num_kv_heads // ctx.tp_size, ctx.model.head_dim
    rt = _op(ctx).forward(_prefill_step())
    assert rt.op_subtype == "prefill"
    assert rt.shape["q_len"] == _SEQ and rt.shape["num_seqs"] == _BS_P
    expected = _flash_prefill_spec(seqlen=_SEQ, bs=_BS_P, n_q=nq, n_kv=nkv,
                                   head_dim=hd, **_bytes(ctx))
    assert _op(ctx).roofline_spec(rt) == expected


def test_decode_resolves_and_spec_matches_formula():
    ctx = _ctx()
    nq, nkv, hd = ctx.model.num_heads // ctx.tp_size, ctx.model.num_kv_heads // ctx.tp_size, ctx.model.head_dim
    rt = _op(ctx).forward(_decode_step())
    assert rt.op_subtype == "decode"
    assert rt.shape["q_len"] == 1 and rt.shape["kv_len"] == _CTXLEN
    expected = _flash_decode_spec(ctx_len=_CTXLEN, bs=_BS_D, n_q=nq, n_kv=nkv,
                                  head_dim=hd, **_bytes(ctx))
    assert _op(ctx).roofline_spec(rt) == expected


def test_mixed_is_single_op_with_synthesized_spec():
    ctx = _ctx()
    nq, nkv, hd = ctx.model.num_heads // ctx.tp_size, ctx.model.num_kv_heads // ctx.tp_size, ctx.model.head_dim
    rt = _op(ctx).forward(_mixed_step())
    assert rt.op_subtype == "mixed"      # 单 op, 不再拆 mixed_prefill/mixed_decode
    pf = _flash_prefill_spec(seqlen=_SEQ, bs=1, n_q=nq, n_kv=nkv, head_dim=hd, **_bytes(ctx))
    dc = _flash_decode_spec(ctx_len=_CTXLEN, bs=_BS_D, n_q=nq, n_kv=nkv, head_dim=hd, **_bytes(ctx))
    assert _op(ctx).roofline_spec(rt) == _add_specs(pf, dc)


def test_single_op_always_active():
    """单 flash op 不再门控 None — 纯 prefill→prefill, 纯 decode→decode, mixed→mixed."""
    ctx = _ctx()
    assert _op(ctx).forward(_prefill_step()).op_subtype == "prefill"
    assert _op(ctx).forward(_decode_step()).op_subtype == "decode"
    assert _op(ctx).forward(_mixed_step()).op_subtype == "mixed"


def test_backend_latency_runs():
    ctx = _ctx()
    rl = RooflineBackend(ctx.hw, ctx.execution_mode)
    for step in (_prefill_step(), _decode_step(), _mixed_step()):
        e = rl.estimate(_op(ctx), _op(ctx).forward(step))
        assert e.latency_s > 0
        assert e.op_kind == "attention"
