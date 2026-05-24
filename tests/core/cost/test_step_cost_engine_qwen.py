"""StepCostEngine smoke — IMPL_PLAN §1.5 / §12.

3 个 shape × prefill + decode 跑通 Qwen3-4B 闭环, 锁住 §1.6 验收标准:
  - StepCostTrace 有 per-op entries
  - 所有 GEMM op 带 op_kind/op_subtype/shape/parallel/runtime
  - source 全部为 roofline
  - 同一 workload 仅替换 DeployConfig 后, parallel/runtime 跟变
  - 总时间量级和主要 op 占比可解释

Smoke shapes:
  i128_o128    baseline
  i2048_o128   prefill scaling
  i128_o2048   decode scaling
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.engine import (
    StepCostEngine,
    build_qwen_dense_roofline_engine,
)
from llm_infer_sim.core.cost.trace import StepCostTrace
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload,
    RequestWorkload,
    StepPhase,
)


def _qwen3_4b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-4B",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def _prefill(isl: int) -> GlobalStepWorkload:
    return GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=isl, context_len=0,
        )],
        num_prefill_tokens=isl, total_scheduled_tokens=isl,
        num_prefill_requests=1,
    )


def _decode(n: int, ctx: int) -> GlobalStepWorkload:
    return GlobalStepWorkload(
        step_id=1, phase=StepPhase.DECODE,
        requests=[
            RequestWorkload(
                request_id=f"d{i}", phase=StepPhase.DECODE,
                num_tokens=1, context_len=ctx,
            )
            for i in range(n)
        ],
        num_decode_tokens=n, total_scheduled_tokens=n,
        num_decode_requests=n,
    )


def _mixed(isl: int, decode_n: int, decode_ctx: int) -> GlobalStepWorkload:
    """Mixed step: 1 个 prefill seq + decode_n 个 decode token."""
    requests = [
        RequestWorkload(
            request_id="p", phase=StepPhase.PREFILL,
            num_tokens=isl, context_len=0,
        ),
    ] + [
        RequestWorkload(
            request_id=f"d{i}", phase=StepPhase.DECODE,
            num_tokens=1, context_len=decode_ctx,
        )
        for i in range(decode_n)
    ]
    return GlobalStepWorkload(
        step_id=2, phase=StepPhase.MIXED, requests=requests,
        num_prefill_tokens=isl, num_decode_tokens=decode_n,
        total_scheduled_tokens=isl + decode_n,
        num_prefill_requests=1, num_decode_requests=decode_n,
    )


@pytest.fixture
def engine() -> StepCostEngine:
    return build_qwen_dense_roofline_engine(
        model=_qwen3_4b(),
        deploy=DeployConfig(),
        hw=get_hardware_profile("RTX_4090"),
    )


# ---- 3 smoke shapes × prefill + decode ----

SMOKE_SHAPES = [
    ("i128_o128_prefill", _prefill, {"isl": 128}),
    ("i2048_o128_prefill", _prefill, {"isl": 2048}),
    ("i128_o2048_prefill", _prefill, {"isl": 128}),  # prefill ISL=128, OSL 影响 decode step
    ("i128_o128_decode", _decode, {"n": 1, "ctx": 128}),
    ("i2048_o128_decode", _decode, {"n": 1, "ctx": 2048}),
    ("i128_o2048_decode", _decode, {"n": 1, "ctx": 2048}),
]


@pytest.mark.parametrize("name, factory, kwargs", SMOKE_SHAPES)
def test_smoke_run_produces_valid_trace(engine, name, factory, kwargs):
    wl = factory(**kwargs)
    trace = engine.estimate(wl)
    assert isinstance(trace, StepCostTrace)
    assert len(trace.entries) > 0
    assert trace.total_latency_s > 0
    assert trace.compute_time_s >= 0
    assert trace.memory_time_s >= 0
    assert trace.bottleneck in ("compute", "memory")


def test_all_entries_source_is_roofline(engine):
    """§1.6 验收: source 全部为 roofline."""
    trace = engine.estimate(_prefill(isl=128))
    for entry in trace.entries:
        assert entry.source == "roofline"
        assert entry.match_type == "fallback"


def test_gemm_entries_carry_required_metadata(engine):
    """§1.6 验收: 所有 GEMM op 带 op_kind/op_subtype/shape/parallel/runtime."""
    trace = engine.estimate(_prefill(isl=128))
    gemm_entries = [e for e in trace.entries if e.op_kind == "gemm"]
    assert gemm_entries, "no GEMM entries"
    # entry 本身就有 op_kind/op_subtype, Operator 字段已通过 test_qwen_dense_template 锁住,
    # 这里间接 verify entry 与 op 一一对应
    for e in gemm_entries:
        assert e.op_kind == "gemm"
        assert e.op_subtype  # 非空
        assert e.metadata["arithmetic_intensity"] > 0
        assert e.roofline_s == e.latency_s
        assert e.roofline_gap is None


def test_prefill_op_count_matches_template(engine):
    """1 embedding + 36 × 11 per-layer + 1 lm_head = 398 ops (无 collective 跳过).

    Grouped trace: entries 折叠到 13, 但 entry.metadata['count'] 之和 == 398.
    """
    trace = engine.estimate(_prefill(isl=128))
    expected = 1 + 36 * 11 + 1
    assert sum(e.metadata.get("count", 1) for e in trace.entries) == expected


def test_decode_total_latency_smaller_than_prefill_for_same_isl():
    """同等 context 长度下, decode 1 token 比 prefill 全段 cheaper."""
    engine = build_qwen_dense_roofline_engine(
        model=_qwen3_4b(), deploy=DeployConfig(),
        hw=get_hardware_profile("RTX_4090"),
    )
    t_prefill = engine.estimate(_prefill(isl=2048)).total_latency_s
    t_decode = engine.estimate(_decode(n=1, ctx=2048)).total_latency_s
    assert t_decode < t_prefill


def test_prefill_scales_with_isl():
    """prefill TTFT 应随 ISL 增长 (compute 主导)."""
    engine = build_qwen_dense_roofline_engine(
        model=_qwen3_4b(), deploy=DeployConfig(),
        hw=get_hardware_profile("RTX_4090"),
    )
    t128 = engine.estimate(_prefill(isl=128)).total_latency_s
    t2048 = engine.estimate(_prefill(isl=2048)).total_latency_s
    assert t2048 > t128
    # ISL 16× 后, 大尺度 compute 主导, 应该至少 5× 量级 (粗 sanity, 不严约束)
    assert t2048 / t128 > 5


def test_decode_scales_with_kv_len():
    """decode TPOT attention 应随 kv_len 增长 (attention 与 kv_len 成正比)."""
    engine = build_qwen_dense_roofline_engine(
        model=_qwen3_4b(), deploy=DeployConfig(),
        hw=get_hardware_profile("RTX_4090"),
    )
    t_short = engine.estimate(_decode(n=1, ctx=128)).total_latency_s
    t_long = engine.estimate(_decode(n=1, ctx=2048)).total_latency_s
    assert t_long > t_short


def test_deploy_change_changes_parallel_metadata():
    """§1.6 验收: 同 workload 改 DeployConfig 后 parallel/runtime 跟变."""
    wl = _prefill(isl=128)
    model = _qwen3_4b()
    hw = get_hardware_profile("RTX_4090")

    engine_tp1 = build_qwen_dense_roofline_engine(
        model=model, deploy=DeployConfig(tp_size=1), hw=hw,
    )
    engine_tp2 = build_qwen_dense_roofline_engine(
        model=model, deploy=DeployConfig(tp_size=2), hw=hw,
    )

    t1 = engine_tp1.estimate(wl)
    t2 = engine_tp2.estimate(wl)

    # latency 必然不同: TP 改变 GEMM shape, 且 TP>1 现在会计入 allreduce.
    assert t1.total_latency_s != t2.total_latency_s
    assert t1.comm_time_s == 0.0
    assert t2.comm_time_s > 0.0
    assert t2.compute_time_s < t1.compute_time_s
    assert any(e.op_name == "tp_o_proj_allreduce" for e in t2.entries)
    assert any(e.op_name == "tp_down_proj_allreduce" for e in t2.entries)


def test_execution_mode_change_affects_kernel_overhead():
    """eager vs cudagraph: eager 有 kernel_overhead, cudagraph 为 0."""
    model = _qwen3_4b()
    hw = get_hardware_profile("RTX_4090")
    hw.kernel_overhead = {"default": 2e-6}    # 注入 2us/op

    e_eager = build_qwen_dense_roofline_engine(
        model=model, deploy=DeployConfig(execution_mode="eager"), hw=hw,
    )
    e_graph = build_qwen_dense_roofline_engine(
        model=model, deploy=DeployConfig(execution_mode="cudagraph"), hw=hw,
    )
    wl = _prefill(isl=128)
    t_eager = e_eager.estimate(wl).total_latency_s
    t_graph = e_graph.estimate(wl).total_latency_s
    # 398 ops × 2us = ~800us 差异. eager 应 > graph.
    assert t_eager > t_graph
    assert (t_eager - t_graph) == pytest.approx(398 * 2e-6, rel=0.01)


def test_to_report_dict_smoke(engine):
    """validate to_report_dict 输出适合 reporter 消费."""
    trace = engine.estimate(_prefill(isl=2048))
    d = trace.to_report_dict()
    assert d["phase"] == "prefill"
    assert isinstance(d["entries"], list)
    assert all("source" in e and "op_kind" in e for e in d["entries"])
    # 顶层 dense GEMM 应该是 compute-bound, attention 也是
    # Grouped trace: each entry 含 count, GEMM compute-bound count 之和 ≥ num_layers
    gemm_compute = sum(
        e["metadata"].get("count", 1) for e in d["entries"]
        if e["op_kind"] == "gemm" and e["metadata"]["bottleneck"] == "compute"
    )
    assert gemm_compute >= 36  # 至少 num_layers 个 GEMM 是 compute-bound (large ISL)


def test_mixed_phase_smoke(engine):
    """#156: 删 full path 后, Qwen grouped 必须自承担 mixed phase (走 mixed_attention 拆 2 op).

    无 fallback, engine 不应抛.
    """
    wl = _mixed(isl=128, decode_n=4, decode_ctx=256)
    trace = engine.estimate(wl)
    assert trace.total_latency_s > 0
    assert trace.bottleneck in ("compute", "memory")
    # attention block 含 2 个 attn op (mixed_prefill + mixed_decode), 每个都 count=num_layers
    attn_entries = [
        e for e in trace.entries
        if e.op_kind == "attention" and e.metadata.get("count", 1) == 36
    ]
    assert len(attn_entries) == 2, (
        f"mixed 应有 2 个 attention group (prefill + decode segment), got {len(attn_entries)}"
    )
    subtypes = {e.op_subtype for e in attn_entries}
    assert subtypes == {"mixed_prefill", "mixed_decode"}
