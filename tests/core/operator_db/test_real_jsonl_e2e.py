"""阶段 3 §3.5 验收: Qwen3-4B 模板 + 真 JSONL → exact hit.

数据源: collector/data/operator_db/RTX_4090/vllm-0.19.1/gemm.jsonl (144 条).
覆盖: qkv_proj / o_proj / gate_up_proj / down_proj / lm_head × M={1,4,16,32,128,512,2048,4096,8192}
      × {eager, cudagraph} × tp=1 × bf16.

验收标准:
  - Qwen3-4B 至少 qkv_proj 或 gate_up_proj exact hit
  - trace 显示 source=operator_db, match_type=exact
  - miss op (无对应 DB shape) 仍 source=roofline
"""
from __future__ import annotations

from pathlib import Path

import pytest

from llm_infer_sim.core.cost.backends.operator_db import OperatorDBBackend
from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.router import CostRouter
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.models.qwen import QwenModelGraphTemplate
from llm_infer_sim.core.operator_db.stores.jsonl import JsonlOperatorStore
from llm_infer_sim.core.operators.factories import (
    AttentionOpFactory, DenseOpFactory,
    EmbeddingOpFactory, FactoryBundle, NormalizationOpFactory,
)
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


_GEMM_JSONL = Path("collector/data/operator_db/RTX_4090/vllm-0.19.1/gemm.jsonl")
pytestmark = pytest.mark.skipif(not _GEMM_JSONL.exists(), reason="real JSONL not present")


def _qwen3_4b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-4B",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def _build_router(*, mode="eager", framework_version="0.19.1"):
    model = _qwen3_4b()
    hw = get_hardware_profile("RTX_4090")
    deploy = DeployConfig(execution_mode=mode, backend_version=framework_version)
    factories = FactoryBundle(
        dense=DenseOpFactory(model, deploy),
        norm=NormalizationOpFactory(model, deploy),
        embedding=EmbeddingOpFactory(model, deploy),
        attention=AttentionOpFactory(model, deploy, hw),
    )
    template = QwenModelGraphTemplate(model)
    rl = RooflineBackend(hw, deploy)
    store = JsonlOperatorStore.from_jsonl(_GEMM_JSONL, hardware="RTX_4090")
    db = OperatorDBBackend(store, roofline=rl)
    router = CostRouter(rl, operator_db=db)
    return template, factories, deploy, router, store


def _prefill_step(model: ModelConfig, deploy: DeployConfig, isl: int) -> StepShape:
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=isl, context_len=0,
        )],
        num_prefill_tokens=isl, total_scheduled_tokens=isl,
        num_prefill_requests=1,
    )
    return StepShape.from_workload(wl, deploy)


def test_jsonl_data_loaded():
    """sanity: 真 JSONL 加载条数."""
    store = JsonlOperatorStore.from_jsonl(_GEMM_JSONL, hardware="RTX_4090")
    assert len(store) >= 140


@pytest.mark.parametrize("isl", [128, 2048])
def test_qwen_qkv_proj_exact_hit(isl):
    """ISL ∈ {128, 2048} prefill 下, qkv_proj 应该 exact hit (DB 有这些 M 值)."""
    template, factories, deploy, router, store = _build_router()
    model = _qwen3_4b()
    step = _prefill_step(model, deploy, isl=isl)
    plan = template.build_step(step, factories)
    trace = router.estimate(plan)
    qkv_entries = [e for e in trace.entries if e.op_subtype == "qkv_proj"]
    assert qkv_entries, "no qkv_proj entries in trace"
    # 所有 36 层 qkv_proj 应该都 hit (M=isl 在 DB sweep 内)
    assert all(e.source == "operator_db" for e in qkv_entries), \
        f"some qkv_proj not hit: {[e.source for e in qkv_entries]}"
    assert all(e.match_type == "exact" for e in qkv_entries)
    e = qkv_entries[0]
    assert e.metadata["case_id"]   # 非空, 表示 DB record provenance
    assert e.metadata["kernel_source"] == "vllm_row_parallel_linear"
    # 实测 vs roofline 对比应存在
    assert e.roofline_s is not None
    assert e.roofline_gap is not None


def test_qwen_gate_up_proj_exact_hit_at_isl_128():
    template, factories, deploy, router, _store = _build_router()
    model = _qwen3_4b()
    step = _prefill_step(model, deploy, isl=128)
    plan = template.build_step(step, factories)
    trace = router.estimate(plan)
    gu = [e for e in trace.entries if e.op_subtype == "gate_up_proj"]
    assert gu, "no gate_up_proj entries"
    assert all(e.source == "operator_db" for e in gu)


def test_attention_ops_remain_roofline():
    """attention 没有真 DB 数据 (collector 没采或不匹配); 应 fallback to roofline."""
    template, factories, deploy, router, _store = _build_router()
    model = _qwen3_4b()
    step = _prefill_step(model, deploy, isl=128)
    plan = template.build_step(step, factories)
    trace = router.estimate(plan)
    attn = [e for e in trace.entries if e.op_kind == "attention"]
    assert attn
    assert all(e.source == "roofline" for e in attn)
    assert all(e.match_type == "fallback" for e in attn)


def test_isl_outside_db_sweep_falls_back_to_roofline():
    """ISL=64 不在 DB sweep ({1,4,16,32,128,...}) 里, qkv_proj 应 fallback to roofline."""
    template, factories, deploy, router, _store = _build_router()
    model = _qwen3_4b()
    step = _prefill_step(model, deploy, isl=64)
    plan = template.build_step(step, factories)
    trace = router.estimate(plan)
    qkv = [e for e in trace.entries if e.op_subtype == "qkv_proj"]
    assert qkv
    assert all(e.source == "roofline" for e in qkv)


def test_cudagraph_mode_also_hits():
    """DB 也覆盖 cudagraph mode, 应该 hit."""
    template, factories, deploy, router, _store = _build_router(mode="cudagraph")
    model = _qwen3_4b()
    step = _prefill_step(model, deploy, isl=128)
    plan = template.build_step(step, factories)
    trace = router.estimate(plan)
    qkv = [e for e in trace.entries if e.op_subtype == "qkv_proj"]
    assert all(e.source == "operator_db" for e in qkv)


def test_wrong_framework_version_misses_all():
    """framework_version=0.20.0 跟 DB 数据 (0.19.1) 不匹配, 所有 GEMM 应 miss → roofline."""
    template, factories, deploy, router, _store = _build_router(framework_version="0.20.0")
    model = _qwen3_4b()
    step = _prefill_step(model, deploy, isl=128)
    plan = template.build_step(step, factories)
    trace = router.estimate(plan)
    gemm = [e for e in trace.entries if e.op_kind == "gemm"]
    assert gemm
    assert all(e.source == "roofline" for e in gemm)


def test_lm_head_decode_token_count_hits_m_1():
    """decode bs=1, lm_head tokens=1 应 hit DB (m=1 在 sweep)."""
    template, factories, deploy, router, _store = _build_router()
    model = _qwen3_4b()
    wl = GlobalStepWorkload(
        step_id=1, phase=StepPhase.DECODE,
        requests=[RequestWorkload(
            request_id="d", phase=StepPhase.DECODE,
            num_tokens=1, context_len=1024,
        )],
        num_decode_tokens=1, total_scheduled_tokens=1,
        num_decode_requests=1,
    )
    step = StepShape.from_workload(wl, deploy)
    plan = template.build_step(step, factories)
    trace = router.estimate(plan)
    head = next(e for e in trace.entries if e.op_subtype == "lm_head")
    assert head.source == "operator_db"
    assert head.match_type == "exact"


def test_trace_mixed_sources_summary():
    """端到端: prefill step trace 应混合 source: GEMM=operator_db (大部分), 其他=roofline."""
    template, factories, deploy, router, _store = _build_router()
    model = _qwen3_4b()
    step = _prefill_step(model, deploy, isl=128)
    plan = template.build_step(step, factories)
    trace = router.estimate(plan)
    sources = [e.source for e in trace.entries]
    assert "operator_db" in sources, "should have at least one DB hit"
    assert "roofline" in sources, "non-GEMM ops still fallback to roofline"
    db_count = sources.count("operator_db")
    # 5 个 GEMM 类型 (qkv/o/gate_up/down/lm_head) × per-layer + lm_head:
    # qkv/o/gate_up/down 各 36 + lm_head 1 = 145, 全 hit; 其他 (norm/elementwise/embedding/attention) = roofline
    # 不严格断言数, 只断言达数量级
    assert db_count >= 100, f"expected ≥100 DB hits, got {db_count}"
