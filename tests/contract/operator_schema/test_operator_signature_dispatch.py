"""Step 2.6 集成测试: operator_to_signature dispatch + Qwen template 端到端.

锁住:
  - dispatch 根据 op_kind 走对应 canonicalizer
  - Qwen3-4B Qwen3Model 生成的 GEMM / attention op 能正确生成 signature
  - 跟同 shape collector case 的 signature 完全一致 (含 stable_hash)
  - 不支持的 op_kind (norm / elementwise / embedding) raise
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.models.qwen3 import Qwen3Model
from llm_infer_sim.core.operator_schema import (
    attention_case_params_to_signature,
    gemm_case_params_to_signature,
    operator_to_signature,
)
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from llm_infer_sim.core.models.config import ModelConfig
from tests.helpers.support import make_model_config
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


def _qwen3_4b() -> ModelConfig:
    return make_model_config(
        name="Qwen3-4B",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def _build_plan(isl=128, deployment=None, runtime=None):
    from llm_infer_sim.core.operators.context import build_operator_context
    model = _qwen3_4b()
    deployment = deployment or DeploymentProfile.flat()
    runtime = runtime or RuntimeProfile.flat()
    hw = get_hardware_profile("RTX_4090")
    ctx = build_operator_context(model, deployment, runtime, hw)
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=isl, context_len=0,
        )],
        num_prefill_tokens=isl, total_scheduled_tokens=isl,
        num_prefill_requests=1,
    )
    step = StepShape.from_workload(wl, runtime.execution.execution_mode)
    return Qwen3Model(model=model, ctx=ctx).forward(step)


def _sig(op, plan):
    """Step-resolved signature (build-once op + forward(runtime))."""
    return op.signature(op.forward(plan.runtime))


def _attn_op(plan):
    """The attention regime active this step (build-once graph carries both)."""
    return next(op for op in plan.ops
                if op.op_kind == "attention" and op.forward(plan.runtime) is not None)


def test_dispatch_qwen_qkv_proj():
    plan = _build_plan(isl=128)
    qkv = next(op for op in plan.ops if op.op_subtype == "qkv_proj")
    sig = _sig(qkv, plan)
    assert sig.op_kind == "gemm"
    assert sig.op_subtype == "qkv_proj"
    shape = dict(sig.shape)
    assert shape["m"] == 128
    # Qwen3-4B: (32 + 2*8) * 128 = 6144
    assert shape["n"] == 6144
    assert shape["k"] == 2560


def test_dispatch_qwen_attention_prefill():
    plan = _build_plan(isl=2048)
    attn = _attn_op(plan)
    sig = _sig(attn, plan)
    assert sig.op_kind == "attention"
    assert sig.op_subtype == "prefill"
    shape = dict(sig.shape)
    assert shape["q_len"] == 2048
    assert shape["kv_len"] == 2048


def test_qwen_qkv_matches_collector_signature():
    """同 shape Qwen3-4B qkv_proj: 模板生成 vs 等价 collector case → 同一 signature."""
    plan = _build_plan(isl=128)
    qkv_op = next(op for op in plan.ops if op.op_subtype == "qkv_proj")
    sig_r = _sig(qkv_op, plan)

    # 等价 collector case (跟 collector/cases/gemm.py::_gemm_case 同一格式).
    collector_case = {
        "op_subtype": "qkv_proj",
        "m": 128, "n": 6144, "k": 2560,
        "dtype": "bf16", "tp": 1, "execution_mode": "eager",
    }
    sig_c = gemm_case_params_to_signature(
        collector_case,
        framework="vllm", framework_version="unknown",
        kernel_source="vllm_row_parallel_linear",
    )
    assert sig_c == sig_r
    assert sig_c.stable_hash() == sig_r.stable_hash()


def test_qwen_attention_matches_collector_signature():
    """同 shape Qwen3-4B attention prefill ISL=2048: 模板 vs collector case → 同 signature."""
    plan = _build_plan(isl=2048)
    attn = _attn_op(plan)
    sig_r = _sig(attn, plan)

    collector_case = {
        "phase": "prefill", "batch_size": 1, "isl": 2048,
        "kv_prefill": 0, "n_decode": 0, "kv_decode": 0,
        "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "dtype": "bf16", "tp": 1, "execution_mode": "eager",
    }
    sig_c = attention_case_params_to_signature(
        collector_case,
        framework="vllm", framework_version="unknown",
        kernel_source="vllm_flash_attn",
        attention_backend="flash_attn",
        kv_dtype="bf16",
        block_size=16,
    )
    assert sig_c == sig_r
    assert sig_c.stable_hash() == sig_r.stable_hash()


def test_dispatch_raises_for_unsupported_kind():
    plan = _build_plan(isl=128)
    norm = next(op for op in plan.ops if op.op_kind == "norm")
    with pytest.raises(ValueError, match="OperatorDB signature contract"):
        operator_to_signature(norm)


def test_dispatch_raises_for_elementwise():
    plan = _build_plan(isl=128)
    add = next(op for op in plan.ops if op.op_subtype == "attn_add")
    with pytest.raises(ValueError, match="OperatorDB signature contract"):
        operator_to_signature(add)


def test_deploy_change_changes_signature():
    """search-ready: 同 op, 改 DeployConfig 后 signature 跟变."""
    plan1 = _build_plan(deployment=DeploymentProfile.flat(tp=1))
    plan2 = _build_plan(deployment=DeploymentProfile.flat(tp=2))
    qkv1 = next(op for op in plan1.ops if op.op_subtype == "qkv_proj")
    qkv2 = next(op for op in plan2.ops if op.op_subtype == "qkv_proj")
    assert _sig(qkv1, plan1) != _sig(qkv2, plan2)


def test_execution_mode_change_changes_signature():
    plan_e = _build_plan(runtime=RuntimeProfile.flat(execution_mode="eager"))
    plan_g = _build_plan(runtime=RuntimeProfile.flat(execution_mode="cudagraph"))
    qkv_e = next(op for op in plan_e.ops if op.op_subtype == "qkv_proj")
    qkv_g = next(op for op in plan_g.ops if op.op_subtype == "qkv_proj")
    assert _sig(qkv_e, plan_e) != _sig(qkv_g, plan_g)


def test_all_qwen_gemm_ops_have_signature():
    """整 Qwen3-4B prefill plan: 所有 op_kind=gemm 的 op 都能生成 signature, 各不相同 across layers/subtypes."""
    plan = _build_plan(isl=128)
    sigs = set()
    for op in plan.ops:
        if op.op_kind == "gemm":
            sig = _sig(op, plan)
            sigs.add(sig.stable_hash())
    # 5 个 subtype × 1 (lm_head 单独) + 4 × num_layers; subtype 内层间不同 (layer_idx 不进 signature)
    # 实际上 layer_idx 不进 signature, 所以 4 subtype × 1 + lm_head = 5 distinct signatures
    assert len(sigs) == 5  # qkv_proj, o_proj, gate_up_proj, down_proj, lm_head
