"""ModelCoreCostModel 阶段 1/2 退出条件 assertion.

不依赖 vLLM —— 直接构造 ProfileBundle 后调 estimate。
"""
from types import SimpleNamespace

import pytest

from llm_infer_sim.core.cost_model.model_core import ModelCoreCostModel
from llm_infer_sim.core.profiles.profile_manager import ProfileManager
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload,
    RequestWorkload,
    StepPhase,
)


def _make_vllm_config(hf, model_id="dummy"):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model=model_id),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1,
        ),
    )


@pytest.fixture
def opt125m_bundle():
    hf = SimpleNamespace(
        model_type="opt",
        num_attention_heads=12, hidden_size=768, num_hidden_layers=12,
        ffn_dim=3072, vocab_size=50272,
    )
    return ProfileManager.from_vllm_config(_make_vllm_config(hf, "opt-125m"))


@pytest.fixture
def qwen3_4b_bundle():
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=2560, num_hidden_layers=36,
        intermediate_size=9728, vocab_size=151936, head_dim=128,
    )
    return ProfileManager.from_vllm_config(_make_vllm_config(hf, "Qwen3-4B"))


def _make_prefill_workload(prefill_tokens=7, num_requests=1):
    return GlobalStepWorkload(
        step_id=1,
        phase=StepPhase.PREFILL,
        requests=[
            RequestWorkload(
                request_id="r0", phase=StepPhase.PREFILL,
                num_tokens=prefill_tokens, context_len=prefill_tokens,
                target_output_len=128,
            ),
        ],
        num_prefill_tokens=prefill_tokens,
        num_decode_tokens=0,
        total_scheduled_tokens=prefill_tokens,
        num_prefill_requests=num_requests,
        num_decode_requests=0,
    )


def _make_decode_workload(num_decode_requests=4, ctx_len=42):
    return GlobalStepWorkload(
        step_id=2,
        phase=StepPhase.DECODE,
        requests=[
            RequestWorkload(
                request_id=f"r{i}", phase=StepPhase.DECODE,
                num_tokens=1, context_len=ctx_len,
                target_output_len=128, generated_tokens=10,
            )
            for i in range(num_decode_requests)
        ],
        num_prefill_tokens=0,
        num_decode_tokens=num_decode_requests,
        total_scheduled_tokens=num_decode_requests,
        num_prefill_requests=0,
        num_decode_requests=num_decode_requests,
    )


# ---------------------------------------------------------------- stage 1


def test_pure_prefill_nonzero_latency(opt125m_bundle):
    """阶段 1 退出: pure prefill 单 step 跑出非零 latency。"""
    cost = ModelCoreCostModel(opt125m_bundle).estimate(_make_prefill_workload(7))
    assert cost["total_time"] > 0
    assert cost["compute_time"] > 0
    assert cost["memory_time"] > 0
    assert cost["comm_time"] == 0    # 阶段 1: 无并行


def test_pure_decode_nonzero_latency(opt125m_bundle):
    """阶段 1 退出: pure decode 单 step 跑出非零 latency。"""
    cost = ModelCoreCostModel(opt125m_bundle).estimate(_make_decode_workload(4, ctx_len=42))
    assert cost["total_time"] > 0
    assert cost["compute_time"] > 0
    assert cost["memory_time"] > 0


def test_deterministic(opt125m_bundle):
    """阶段 1 退出: 同一份输入两次跑出 deterministic 结果。"""
    model = ModelCoreCostModel(opt125m_bundle)
    w = _make_prefill_workload(7)
    a = model.estimate(w)
    b = model.estimate(w)
    assert a["total_time"] == b["total_time"]
    assert a["compute_time"] == b["compute_time"]
    assert a["memory_time"] == b["memory_time"]


def test_breakdown_three_columns_nonzero(opt125m_bundle):
    """阶段 1 退出: breakdown 三栏比例非零 (comm 阶段 1 = 0 是设计)。"""
    cost = ModelCoreCostModel(opt125m_bundle).estimate(_make_prefill_workload(16))
    assert cost["compute_time"] > 0
    assert cost["memory_time"] > 0
    # comm 在阶段 1 必须 = 0 (单卡)
    assert cost["comm_time"] == 0


def test_empty_workload(opt125m_bundle):
    """空 step 路径: total_scheduled_tokens=0 → 全 0."""
    w = GlobalStepWorkload(step_id=0, phase=StepPhase.DECODE)
    cost = ModelCoreCostModel(opt125m_bundle).estimate(w)
    assert cost["total_time"] == 0
    assert cost["per_op"] == []


# ---------------------------------------------------------------- stage 2


def _ops_by_name(per_op, name):
    return [r for r in per_op if r["name"] == name]


def test_qwen3_fused_gate_up_and_down(qwen3_4b_bundle):
    """阶段 3 起 SwiGLU 走 MergedColumnParallelLinear: gate_up_proj + down_proj。"""
    cost = ModelCoreCostModel(qwen3_4b_bundle).estimate(_make_prefill_workload(7))
    per_op = cost["per_op"]
    assert _ops_by_name(per_op, "gate_up_proj"), "missing fused gate_up_proj (阶段 3)"
    assert _ops_by_name(per_op, "down_proj"), "missing down_proj"
    # 旧的 gate_proj / up_proj 不应该再存在
    assert not _ops_by_name(per_op, "gate_proj"), "gate_proj should be fused into gate_up_proj"
    assert not _ops_by_name(per_op, "up_proj"), "up_proj should be fused into gate_up_proj"


def test_qwen3_gqa_qkv_fusion(qwen3_4b_bundle):
    """阶段 3 起 Qwen3 GQA 走 fused QKVParallelLinear。

    Qwen3-4B: num_heads=32, num_kv_heads=8, head_dim=128.
    fused QKV 输出 dim = (32 + 2*8) * 128 = 6144 per tp。
    """
    cost = ModelCoreCostModel(qwen3_4b_bundle).estimate(_make_prefill_workload(7))
    per_op = cost["per_op"]
    qkv_layer0 = next(r for r in per_op if r["name"] == "qkv_proj" and r["scope"] == "layer0")
    # FLOPs = 2 * tokens * hidden * (n_q + 2*n_kv) * head_dim
    #       = 2 * 7 * 2560 * (32 + 16) * 128 = 2 * 7 * 2560 * 6144
    expected = 2 * 7 * 2560 * (32 + 2 * 8) * 128
    assert qkv_layer0["flops"] == expected, f"qkv_proj flops {qkv_layer0['flops']} != {expected}"
    # 旧 q/k/v_proj 不应再存在
    assert not _ops_by_name(per_op, "q_proj"), "q_proj should be fused into qkv_proj"


def test_qwen3_per_op_count_after_fusion(qwen3_4b_bundle):
    """阶段 3 起每层 op 集合 (相比阶段 2 减 4, 加 1 RoPE = 净减 3):
    attn_norm + qkv_proj + rope + fused_attention + o_proj + attn_add
    + mlp_norm + gate_up_proj + mlp_act + down_proj + mlp_add = 11 ops
    """
    cost = ModelCoreCostModel(qwen3_4b_bundle).estimate(_make_prefill_workload(7))
    per_op = cost["per_op"]
    layer0_ops = [r for r in per_op if r["scope"] == "layer0"]
    assert len(layer0_ops) == 11, f"layer0 has {len(layer0_ops)} ops: {[r['name'] for r in layer0_ops]}"


def test_opt_qkv_fusion_mha(opt125m_bundle):
    """opt-125m 走 MHA, 阶段 3 起也走 fused qkv。"""
    cost = ModelCoreCostModel(opt125m_bundle).estimate(_make_prefill_workload(7))
    per_op = cost["per_op"]
    qkv = next(r for r in per_op if r["name"] == "qkv_proj" and r["scope"] == "layer0")
    # MHA: num_q_heads == num_kv_heads, fused output dim = 3 * num_heads * head_dim
    assert qkv["flops"] > 0
    # rope 算子也应该在
    rope_ops = [r for r in per_op if r["name"] == "rope" and r["scope"] == "layer0"]
    assert len(rope_ops) == 1


def test_total_time_self_consistent(qwen3_4b_bundle):
    """sum(per_op t_total) ≈ total_time (允许少量浮点误差; per_op 走 op-level
    roofline, total_time 走 layer-level roofline, 两者数学上一般不严格相等
    但 opt/qwen3 全 memory-bound 时 sum_op_max ≈ sum_layer_max).
    """
    cost = ModelCoreCostModel(qwen3_4b_bundle).estimate(_make_decode_workload(1, ctx_len=42))
    sum_op = sum(r["t_total"] for r in cost["per_op"])
    # 单 op roofline = max(t_compute, t_memory), 求和
    # cost["total_time"] 是 sum(layer roofline) + sum(emb/lmh op roofline)
    # 两者只在所有 layer 内 op 全 memory-bound 且无 compute-bound 时相等
    # opt/qwen 在 batch=1 decode 几乎一定全 memory-bound, 所以这里相等
    assert abs(sum_op - cost["total_time"]) < 1e-9


# ---------------------------------------------------------------- stage 2 收尾
# §4.7.1 estimate dict 输出补 attention_time / linear_time / moe_time / bottleneck


def test_estimate_dict_has_aggregate_fields(qwen3_4b_bundle):
    """阶段 2 收尾: estimate dict 必须包含 4 个聚合字段 (§4.7.1)."""
    cost = ModelCoreCostModel(qwen3_4b_bundle).estimate(_make_prefill_workload(7))
    for key in ("attention_time", "linear_time", "moe_time", "bottleneck"):
        assert key in cost, f"missing key {key!r}"


def test_attention_time_matches_per_op_sum(qwen3_4b_bundle):
    """attention_time = sum(per_op[i].t_total) for category=='attention'."""
    cost = ModelCoreCostModel(qwen3_4b_bundle).estimate(_make_prefill_workload(7))
    expected = sum(
        r["t_total"] for r in cost["per_op"] if r["category"] == "attention"
    )
    assert cost["attention_time"] == expected
    assert cost["attention_time"] > 0  # Qwen3-4B 36 层 attention 必非零


def test_linear_time_matches_per_op_sum(qwen3_4b_bundle):
    """linear_time = sum(per_op[i].t_total) for category=='matmul'.

    matmul 包含 q/k/v_proj, o_proj, gate/up/down_proj, lm_head 等所有 GEMM。
    """
    cost = ModelCoreCostModel(qwen3_4b_bundle).estimate(_make_prefill_workload(7))
    expected = sum(
        r["t_total"] for r in cost["per_op"] if r["category"] == "matmul"
    )
    assert cost["linear_time"] == expected
    # 4B dense GEMM 是 step latency 主体, 远大于 attention
    assert cost["linear_time"] > cost["attention_time"]


def test_moe_time_zero_for_dense(qwen3_4b_bundle):
    """阶段 2 dense 模型: moe_time = 0 (Qwen3-4B 不是 MoE)."""
    cost = ModelCoreCostModel(qwen3_4b_bundle).estimate(_make_prefill_workload(7))
    assert cost["moe_time"] == 0.0


def test_bottleneck_value_in_known_set(qwen3_4b_bundle):
    """bottleneck 必须 ∈ {compute, memory, communication, unknown}."""
    cost = ModelCoreCostModel(qwen3_4b_bundle).estimate(_make_prefill_workload(7))
    assert cost["bottleneck"] in ("compute", "memory", "communication", "unknown")


def test_bottleneck_memory_for_4b_single_card(qwen3_4b_bundle):
    """Qwen3-4B 单卡 batch=1: 必然 memory-bound (权重读取 ~2.4ms 主导)."""
    cost = ModelCoreCostModel(qwen3_4b_bundle).estimate(_make_decode_workload(1, ctx_len=42))
    assert cost["bottleneck"] == "memory"
    assert cost["memory_time"] > cost["compute_time"]


def test_empty_workload_aggregate_fields(opt125m_bundle):
    """空 step 路径: 4 个聚合字段也要存在且为 0 / unknown."""
    w = GlobalStepWorkload(step_id=0, phase=StepPhase.DECODE)
    cost = ModelCoreCostModel(opt125m_bundle).estimate(w)
    assert cost["attention_time"] == 0.0
    assert cost["linear_time"] == 0.0
    assert cost["moe_time"] == 0.0
    assert cost["bottleneck"] == "unknown"
