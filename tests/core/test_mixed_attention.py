"""mixed step (prefill+decode 同 step) attention 拆解.

链路: QwenModelGraphTemplate._attention_ops 在 mixed/chunked_prefill phase 拆 2 个
Attention op (prefill segment + decode segment), 各自走 flash_attention_*_formula
helper 算公式.

阶段 3d 范围 (split_kernels only):
  - prefill 段公式 = flash_attention_prefill_formula
  - decode 段公式 = flash_attention_decode_formula
  - 总成本 = sum (split_kernels) — CostRouter 不另算 sync overhead

未实现:
  - unified_ragged (FA varlen / FlashInfer 单 kernel ragged)  → Stage 6 ModuleProfile
  - chunked_prefill_interleaved / decode_priority_prefill_append → 后续阶段
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.engine import build_qwen_dense_roofline_engine
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


def _qwen3_4b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-4B",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def _build_engine():
    return build_qwen_dense_roofline_engine(
        _qwen3_4b(), DeployConfig(), get_hardware_profile("RTX_4090"),
    )


def _mixed_step(*, isl=200, n_decode=4, ctx_decode=512) -> StepShape:
    requests = [
        RequestWorkload(
            request_id="p0", phase=StepPhase.PREFILL,
            num_tokens=isl, context_len=0,
        ),
    ] + [
        RequestWorkload(
            request_id=f"d{i}", phase=StepPhase.DECODE,
            num_tokens=1, context_len=ctx_decode,
        )
        for i in range(n_decode)
    ]
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.MIXED, requests=requests,
        num_prefill_tokens=isl, num_decode_tokens=n_decode,
        total_scheduled_tokens=isl + n_decode,
        num_prefill_requests=1, num_decode_requests=n_decode,
    )
    return StepShape.from_workload(wl, DeployConfig())


# ---- StepShape 支持 mixed ----

def test_step_shape_accepts_mixed_phase():
    step = _mixed_step()
    assert step.phase == "mixed"
    assert step.num_prefill_tokens == 200
    assert step.num_decode_requests == 4
    assert step.max_prefill_seqlen == 200
    assert step.avg_decode_context_len == 512


# ---- mixed_attention split kernels ----

def test_mixed_attention_returns_two_ops_for_mixed_step():
    engine = _build_engine()
    step = _mixed_step()
    ops = engine.template._attention_ops(0, step, engine.template.ctx)
    assert len(ops) == 2
    subtypes = sorted(op.op_subtype for op in ops)
    assert subtypes == ["mixed_decode", "mixed_prefill"]


def test_mixed_attention_tags_both_with_mixed():
    engine = _build_engine()
    ops = engine.template._attention_ops(0, _mixed_step(), engine.template.ctx)
    for op in ops:
        assert "mixed" in op.tags


def test_mixed_attention_prefill_segment_matches_standalone_prefill():
    """mixed.prefill 段公式 = pure prefill StepShape.attention 公式 (用同 isl)."""
    engine = _build_engine()
    step_mixed = _mixed_step(isl=200, n_decode=4)
    ops_mixed = engine.template._attention_ops(0, step_mixed, engine.template.ctx)
    pf_op = next(op for op in ops_mixed if op.op_subtype == "mixed_prefill")

    # pure prefill ref
    pf_only_wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="p", phase=StepPhase.PREFILL,
            num_tokens=200, context_len=0,
        )],
        num_prefill_tokens=200, total_scheduled_tokens=200,
        num_prefill_requests=1,
    )
    ref_step = StepShape.from_workload(pf_only_wl, engine.deploy)
    ref_op = engine.template._attention_ops(0, ref_step, engine.template.ctx)[0]
    assert pf_op.roofline_spec().flops == ref_op.roofline_spec().flops
    assert pf_op.roofline_spec().mem_bytes == ref_op.roofline_spec().mem_bytes


def test_mixed_attention_decode_segment_matches_standalone_decode():
    """mixed.decode 段公式 = pure decode StepShape.attention 公式 (同 n_decode, ctx)."""
    engine = _build_engine()
    step_mixed = _mixed_step(isl=200, n_decode=4, ctx_decode=512)
    ops_mixed = engine.template._attention_ops(0, step_mixed, engine.template.ctx)
    dc_op = next(op for op in ops_mixed if op.op_subtype == "mixed_decode")

    # pure decode ref
    dc_only_wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.DECODE,
        requests=[RequestWorkload(
            request_id=f"d{i}", phase=StepPhase.DECODE,
            num_tokens=1, context_len=512,
        ) for i in range(4)],
        num_decode_tokens=4, total_scheduled_tokens=4,
        num_decode_requests=4,
    )
    ref_step = StepShape.from_workload(dc_only_wl, engine.deploy)
    ref_op = engine.template._attention_ops(0, ref_step, engine.template.ctx)[0]
    assert dc_op.roofline_spec().flops == ref_op.roofline_spec().flops
    assert dc_op.roofline_spec().load_kv_cache == ref_op.roofline_spec().load_kv_cache


def test_mixed_attention_no_prefill_drops_prefill_op():
    """mixed step 没有 prefill tokens (n=0) → 只返 decode op."""
    engine = _build_engine()
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.MIXED,
        requests=[RequestWorkload(
            request_id=f"d{i}", phase=StepPhase.DECODE,
            num_tokens=1, context_len=512,
        ) for i in range(4)],
        num_prefill_tokens=0, num_decode_tokens=4,
        total_scheduled_tokens=4,
        num_prefill_requests=0, num_decode_requests=4,
    )
    step = StepShape.from_workload(wl, engine.deploy)
    ops = engine.template._attention_ops(0, step, engine.template.ctx)
    assert len(ops) == 1
    assert ops[0].op_subtype == "mixed_decode"


def test_mixed_attention_no_decode_drops_decode_op():
    """mixed step 没有 decode (但 phase=mixed) → 只返 prefill op."""
    engine = _build_engine()
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.MIXED,
        requests=[RequestWorkload(
            request_id="p", phase=StepPhase.PREFILL,
            num_tokens=200, context_len=0,
        )],
        num_prefill_tokens=200, num_decode_tokens=0,
        total_scheduled_tokens=200,
        num_prefill_requests=1, num_decode_requests=0,
    )
    step = StepShape.from_workload(wl, engine.deploy)
    ops = engine.template._attention_ops(0, step, engine.template.ctx)
    assert len(ops) == 1
    assert ops[0].op_subtype == "mixed_prefill"


def test_attention_ops_returns_single_for_pure_phase():
    """template._attention_ops 在 prefill/decode 纯 phase 返 list 1 op (不再拒)."""
    engine = _build_engine()
    pf_only_wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="p", phase=StepPhase.PREFILL,
            num_tokens=128, context_len=0,
        )],
        num_prefill_tokens=128, total_scheduled_tokens=128,
        num_prefill_requests=1,
    )
    step = StepShape.from_workload(pf_only_wl, engine.deploy)
    ops = engine.template._attention_ops(0, step, engine.template.ctx)
    assert len(ops) == 1
    assert ops[0].op_subtype == "prefill"


def test_mixed_attention_accepts_chunked_prefill_phase():
    """chunked_prefill 跟 mixed 同等对待."""
    engine = _build_engine()
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.CHUNKED_PREFILL,
        requests=[
            RequestWorkload(
                request_id="p", phase=StepPhase.PREFILL,
                num_tokens=200, context_len=0,
            ),
        ],
        num_prefill_tokens=200, total_scheduled_tokens=200,
        num_prefill_requests=1,
    )
    step = StepShape.from_workload(wl, engine.deploy)
    assert step.phase == "chunked_prefill"
    # decode 段没有,只返 prefill
    ops = engine.template._attention_ops(0, step, engine.template.ctx)
    assert len(ops) == 1
    assert ops[0].op_subtype == "mixed_prefill"
