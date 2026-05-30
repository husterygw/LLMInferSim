"""engine path invariants: StepCostEngine.estimate runs the static-contract
forward()/StepOpPlan path for every model family (Qwen dense, Qwen MoE, DeepSeek
MLA) and the per-layer op multiplicity (sum of op.count) is preserved. Numerical
correctness is locked by the per-op static-contract equivalence tests + bench;
this guards the engine wiring (forward path taken, count preserved, latency > 0).
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.engine import (
    build_deepseek_roofline_engine, build_qwen_roofline_engine,
)
from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from tests.helpers.support import make_model_config
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)

_DENSE = make_model_config(name="Qwen3-4B", hidden_dim=2560, num_heads=32, num_kv_heads=8,
                     head_dim=128, ffn_dim=9728, num_layers=36, vocab_size=151936)
_MOE = make_model_config(name="Qwen3-30B-A3B", hidden_dim=2048, num_heads=32, num_kv_heads=4,
                   head_dim=128, ffn_dim=6144, num_layers=48, vocab_size=151936,
                   is_moe=True, num_experts=128, num_activated_experts=8, expert_dim=768,
                   moe_layer_freq=1, first_moe_layer=0)
_DEEPSEEK = make_model_config(name="DeepSeek-V3", hidden_dim=7168, num_heads=128, num_kv_heads=128,
                        head_dim=56, ffn_dim=18432, num_layers=61, vocab_size=129280,
                        is_moe=True, num_experts=256, num_activated_experts=8, expert_dim=2048,
                        num_shared_experts=1, moe_layer_freq=1, first_moe_layer=3,
                        kv_lora_rank=512, kv_latent_dim=576, qk_nope_head_dim=128,
                        v_head_dim=128, rope_head_dim=64, q_lora_rank=1536)


def _pf(isl):
    return GlobalStepWorkload(step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(request_id="p", phase=StepPhase.PREFILL,
                                  num_tokens=isl, context_len=0)],
        num_prefill_tokens=isl, total_scheduled_tokens=isl, num_prefill_requests=1)


def _dec(n):
    return GlobalStepWorkload(step_id=1, phase=StepPhase.DECODE,
        requests=[RequestWorkload(request_id=f"d{i}", phase=StepPhase.DECODE,
                                  num_tokens=1, context_len=512, generated_tokens=8)
                  for i in range(n)],
        num_decode_tokens=n, total_scheduled_tokens=n, num_decode_requests=n)


# expected total per-layer op multiplicity (sum of op.count) per case — frozen
# structural snapshot of the model graph.
_CASES = [
    ("qwen-dense-tp1", build_qwen_roofline_engine, _DENSE, DeploymentProfile.flat(tp=1), RuntimeProfile.flat(execution_mode="cudagraph"), "cudagraph", _pf(2048), 398),
    ("qwen-dense-tp4", build_qwen_roofline_engine, _DENSE, DeploymentProfile.flat(tp=4), RuntimeProfile.flat(execution_mode="cudagraph"), "cudagraph", _dec(8), 470),
    # MoE comm 折进 moe_dispatch_post (AIC 对齐) 后每 MoE 层少 1 个独立 allreduce op:
    #   qwen-moe 722→674 (-48 层), deepseek 1265→1207 (-58 层).
    ("qwen-moe-tp4ep4", build_qwen_roofline_engine, _MOE, DeploymentProfile.flat(tp=4, ep=4, moe_ep=4), RuntimeProfile.flat(execution_mode="cudagraph"), "cudagraph", _pf(2048), 674),
    ("deepseek-tp8-prefill", build_deepseek_roofline_engine, _DEEPSEEK, DeploymentProfile.flat(tp=8), RuntimeProfile.flat(), "eager", _pf(2048), 1207),
    ("deepseek-tp8-decode", build_deepseek_roofline_engine, _DEEPSEEK, DeploymentProfile.flat(tp=8), RuntimeProfile.flat(), "eager", _dec(8), 1207),
]


@pytest.mark.parametrize("label,builder,model,deployment,runtime,execution_mode,wl,expected_count", _CASES, ids=[c[0] for c in _CASES])
def test_engine_runs_forward_path(label, builder, model, deployment, runtime, execution_mode, wl, expected_count):
    eng = builder(model, deployment, runtime, get_hardware_profile("RTX_4090"))
    step = StepShape.from_workload(wl, execution_mode)
    # production engine routes through the static-contract forward()/StepOpPlan path
    assert isinstance(eng.model.forward(step), StepOpPlan)
    trace = eng.estimate(wl)
    assert trace.total_latency_s > 0
    # per-layer op multiplicity preserved (structural invariant)
    assert sum(e.metadata.get("count", 1) for e in trace.entries) == expected_count
