"""V3 §4.2 StepShape 单测 — 阶段 1 Step 1.2.

锁住:
  - from_workload 正确处理 prefill / decode
  - MIXED / CHUNKED_PREFILL 抛 NotImplementedError
  - execution_mode 来自 DeployConfig
  - frozen
"""
from __future__ import annotations

import dataclasses

import pytest

from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload,
    RequestWorkload,
    StepPhase,
)


def _prefill_workload(isl: int = 128, bs: int = 1) -> GlobalStepWorkload:
    requests = [
        RequestWorkload(
            request_id=f"r{i}",
            phase=StepPhase.PREFILL,
            num_tokens=isl,
            context_len=0,
        )
        for i in range(bs)
    ]
    return GlobalStepWorkload(
        step_id=0,
        phase=StepPhase.PREFILL,
        requests=requests,
        num_prefill_tokens=isl * bs,
        num_decode_tokens=0,
        total_scheduled_tokens=isl * bs,
        num_prefill_requests=bs,
        num_decode_requests=0,
    )


def _decode_workload(n: int = 4, ctx: int = 1024) -> GlobalStepWorkload:
    requests = [
        RequestWorkload(
            request_id=f"d{i}",
            phase=StepPhase.DECODE,
            num_tokens=1,
            context_len=ctx,
        )
        for i in range(n)
    ]
    return GlobalStepWorkload(
        step_id=1,
        phase=StepPhase.DECODE,
        requests=requests,
        num_prefill_tokens=0,
        num_decode_tokens=n,
        total_scheduled_tokens=n,
        num_prefill_requests=0,
        num_decode_requests=n,
    )


def test_from_prefill_workload():
    wl = _prefill_workload(isl=2048, bs=1)
    deploy = DeployConfig(execution_mode="eager")
    shape = StepShape.from_workload(wl, deploy)
    assert shape.step_id == 0
    assert shape.phase == "prefill"
    assert shape.total_tokens == 2048
    assert shape.num_prefill_tokens == 2048
    assert shape.num_decode_tokens == 0
    assert shape.num_prefill_requests == 1
    assert shape.max_prefill_seqlen == 2048
    assert shape.execution_mode == "eager"


def test_from_decode_workload():
    wl = _decode_workload(n=8, ctx=1024)
    deploy = DeployConfig(execution_mode="cudagraph")
    shape = StepShape.from_workload(wl, deploy)
    assert shape.phase == "decode"
    assert shape.total_tokens == 8
    assert shape.num_decode_tokens == 8
    assert shape.num_decode_requests == 8
    assert shape.avg_decode_context_len == 1024
    assert shape.max_context_len == 1024
    assert shape.execution_mode == "cudagraph"


def test_mixed_phase_now_supported():
    """3d 起 StepShape 接受 mixed phase (prefill+decode 同 step)."""
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.MIXED,
        num_prefill_tokens=128, num_decode_tokens=4,
        total_scheduled_tokens=132,
        num_prefill_requests=1, num_decode_requests=4,
    )
    deploy = DeployConfig()
    shape = StepShape.from_workload(wl, deploy)
    assert shape.phase == "mixed"
    assert shape.num_prefill_tokens == 128
    assert shape.num_decode_requests == 4


def test_chunked_prefill_now_supported():
    """3d 起 chunked_prefill 也接受 (跟 mixed 同语义,只是 phase label 不同)."""
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.CHUNKED_PREFILL,
        num_prefill_tokens=128, total_scheduled_tokens=128,
        num_prefill_requests=1,
    )
    deploy = DeployConfig()
    shape = StepShape.from_workload(wl, deploy)
    assert shape.phase == "chunked_prefill"


def test_unknown_phase_raises():
    """非 prefill/decode/mixed/chunked_prefill 仍 raise."""
    wl = GlobalStepWorkload(step_id=0, phase="weird")
    deploy = DeployConfig()
    with pytest.raises(NotImplementedError):
        StepShape.from_workload(wl, deploy)


def test_frozen():
    wl = _prefill_workload()
    shape = StepShape.from_workload(wl, DeployConfig())
    with pytest.raises(dataclasses.FrozenInstanceError):
        shape.phase = "decode"  # type: ignore[misc]


def test_execution_mode_follows_deploy():
    """search-ready 验证: 同 workload 不同 DeployConfig 拿到不同 StepShape."""
    wl = _prefill_workload()
    s_eager = StepShape.from_workload(wl, DeployConfig(execution_mode="eager"))
    s_graph = StepShape.from_workload(wl, DeployConfig(execution_mode="cudagraph"))
    assert s_eager.execution_mode == "eager"
    assert s_graph.execution_mode == "cudagraph"
    assert s_eager != s_graph


def test_graph_capture_and_padded_default_none():
    wl = _prefill_workload()
    shape = StepShape.from_workload(wl, DeployConfig())
    assert shape.graph_capture_size is None
    assert shape.padded_tokens is None
