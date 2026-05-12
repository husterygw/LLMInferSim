"""阶段 5-β: MoE cost 公式数字一致性 (详设 §4.7.1 + §4.7.1a (3))。

固化以下关键正确性 (手算 vs layer_builder 实际输出对照):
  1. routed_experts.flops = tokens × top_k × 3 × 2 × h × expert_dim_per_device
  2. routed_experts.mem_bytes 的 weight 部分只读 top_k 个 expert (不读全部 num_experts)
     —— 这是 FusedMoE 语义的关键, 也是 MoE 收益的来源
  3. routed_experts.mem_bytes 不含中间激活的 HBM 读写 (FusedMoE 不写中间, §4.7.1a (3))
  4. moe_gate.flops = 2 × tokens × h × num_experts (路由全连接)
  5. ep=1 + tp>1: 自动注入 routed_expert_allreduce comm op
"""
from types import SimpleNamespace

import pytest

from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
from llm_infer_sim.core.cost_model.layer_builder import moe_layer_time


@pytest.fixture
def qwen3_30b_a3b_bundle():
    hf = SimpleNamespace(
        model_type="qwen3_moe",
        num_attention_heads=32, num_key_value_heads=4,
        hidden_size=2048, num_hidden_layers=48,
        intermediate_size=6144, vocab_size=151936, head_dim=128,
        num_experts=128, num_experts_per_tok=8,
        moe_intermediate_size=768, mlp_only_layers=[],
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="Qwen/Qwen3-30B-A3B"),
        parallel_config=SimpleNamespace(tensor_parallel_size=2, data_parallel_size=1),
    )
    return extract_profile_bundle(vc)


def _find_op(lr, name):
    for op in lr.ops:
        if op.name == name:
            return op
    raise AssertionError(f"op {name!r} not found in layer ops {[o.name for o in lr.ops]}")


def test_routed_experts_flops_uses_top_k_not_num_experts(qwen3_30b_a3b_bundle):
    """每 token 只对 top_k 个 expert 算 FFN,不是全部 128 个。"""
    b = qwen3_30b_a3b_bundle
    m, deploy, hw = b.model, b.deploy, b.hw
    tokens = 4
    lr = moe_layer_time(0, "decode", tokens, 128, m, deploy, hw)
    op = _find_op(lr, "routed_experts")

    expert_dim_per_device = m.expert_dim // deploy.tp  # 768 / 2 = 384
    expected_flops = (
        tokens * m.num_activated_experts * 3 * 2 * m.hidden_dim * expert_dim_per_device
    )
    assert op.flops == expected_flops, f"expected {expected_flops}, got {op.flops}"

    # 防御性检查: 如果误用 num_experts 会差 (num_experts / top_k = 16) 倍
    wrong_full_experts = tokens * m.num_experts * 3 * 2 * m.hidden_dim * expert_dim_per_device
    assert op.flops != wrong_full_experts


def test_routed_experts_weight_read_decode_single_token_equals_topk(qwen3_30b_a3b_bundle):
    """边界: tokens=1 + skew=0 时 distinct_experts == top_k (decode 严格成立)。

    这是 MoE 收益的根本来源在 decode 边界的体现: 单 token 时每 step 真的只读
    top_k 个 expert 的权重。tokens>1 时见 test_routed_experts_weight_read_scales_with_distinct。
    """
    b = qwen3_30b_a3b_bundle
    m, deploy, hw = b.model, b.deploy, b.hw
    tokens = 1
    lr = moe_layer_time(0, "decode", tokens, 128, m, deploy, hw)
    op = _find_op(lr, "routed_experts")

    expert_dim_per_device = m.expert_dim // deploy.tp  # 384
    expected_weight_read = int(
        m.num_activated_experts * 3 * m.hidden_dim * expert_dim_per_device * deploy.w_byte
    )
    expected_act = 2 * tokens * m.hidden_dim * deploy.a_byte
    assert op.mem_bytes == expected_weight_read + expected_act


def test_routed_experts_weight_read_scales_with_distinct(qwen3_30b_a3b_bundle):
    """阶段 5-δ 关键: tokens>1 时 weight_read 用 distinct_experts (coupon collector),
    而非硬编码 top_k。distinct(T=4) ≈ 29.12, distinct(T=128) ≈ 128。
    """
    from llm_infer_sim.core.cost_model.moe_routing import estimate_distinct_experts
    b = qwen3_30b_a3b_bundle
    m, deploy, hw = b.model, b.deploy, b.hw
    expert_dim_per_device = m.expert_dim // deploy.tp

    for tokens in (4, 128):
        lr = moe_layer_time(0, "prefill", tokens, 128, m, deploy, hw)
        op = _find_op(lr, "routed_experts")
        distinct = estimate_distinct_experts(
            tokens, m.num_activated_experts, m.num_experts, skew=0.0,
        )
        expected_weight = int(
            distinct * 3 * m.hidden_dim * expert_dim_per_device * deploy.w_byte
        )
        expected_act = 2 * tokens * m.hidden_dim * deploy.a_byte
        assert op.mem_bytes == expected_weight + expected_act, (
            f"tokens={tokens}: got {op.mem_bytes}, expected {expected_weight + expected_act} "
            f"(distinct={distinct:.2f})"
        )


def test_routed_experts_no_intermediate_hbm_write(qwen3_30b_a3b_bundle):
    """FusedMoE 中间激活不写 HBM (§4.7.1a (3))。

    用 tokens=1 (distinct=top_k, 公式确定) 比对: 实际 mem_bytes 应严格等于
    weight + Q/O,**不含**任何 `intermediate (m_e × 2N × 2 × a_byte)` 项。
    """
    b = qwen3_30b_a3b_bundle
    m, deploy, hw = b.model, b.deploy, b.hw
    tokens = 1
    lr = moe_layer_time(0, "decode", tokens, 128, m, deploy, hw)
    op = _find_op(lr, "routed_experts")

    expert_dim_per_device = m.expert_dim // deploy.tp
    weight = m.num_activated_experts * 3 * m.hidden_dim * expert_dim_per_device * deploy.w_byte
    qo_act = 2 * tokens * m.hidden_dim * deploy.a_byte
    # 严格等价 (无 intermediate)
    assert op.mem_bytes == int(weight + qo_act)

    # 防御性: 如果谁误加了 naive intermediate, mem_bytes 会变大
    m_e = max(1, tokens * m.num_activated_experts // m.num_experts)
    naive_intermediate = 2 * m_e * 2 * expert_dim_per_device * deploy.a_byte
    assert op.mem_bytes < weight + qo_act + naive_intermediate


def test_moe_gate_flops(qwen3_30b_a3b_bundle):
    """Router gate 是全连接, flops = 2 × tokens × hidden × num_experts。"""
    b = qwen3_30b_a3b_bundle
    m, deploy, hw = b.model, b.deploy, b.hw
    tokens = 4
    lr = moe_layer_time(0, "decode", tokens, 128, m, deploy, hw)
    op = _find_op(lr, "moe_gate")
    assert op.flops == 2 * tokens * m.hidden_dim * m.num_experts


def test_routed_expert_allreduce_present_under_ep1_tp2(qwen3_30b_a3b_bundle):
    """ep=1 + tp>1 (Row Parallel 风格), routed expert 后必须插 allreduce。"""
    b = qwen3_30b_a3b_bundle
    assert b.deploy.ep == 1 and b.deploy.tp == 2
    lr = moe_layer_time(0, "decode", tokens=4, ctx_len=128,
                        model=b.model, deploy=b.deploy, hw=b.hw)
    op = _find_op(lr, "routed_expert_allreduce")
    assert op.op_category == "communication"
    assert op.comm_type == "allreduce"
    # comm_bytes = tokens × hidden × a_byte
    assert op.comm_bytes == 4 * b.model.hidden_dim * b.deploy.a_byte


def test_sizing_per_rank_storage_matches_observed_30gb(qwen3_30b_a3b_bundle):
    """sizing.per_rank_param_bytes 应该跟实测 30.5 GB / rank 接近 (±10%)。

    sizing 算的是 *存储* (TP 切 intermediate_size, 每 rank 仍持有全部 128 expert),
    不是 *运行时读取* (那是 routed_experts.mem_bytes 的 weight 部分, 只 top_k)。
    两个量不同维度, 这个测试只验存储口径。
    """
    from llm_infer_sim.core.profiles.sizing import per_rank_param_bytes
    b = qwen3_30b_a3b_bundle
    per_rank = per_rank_param_bytes(b.model, b.deploy.w_byte, b.deploy.tp)
    per_rank_gb = per_rank / 1e9
    assert 27.0 < per_rank_gb < 33.0, f"got {per_rank_gb:.2f} GB"


def test_routed_experts_storage_vs_read_ratio_matches_topk_for_single_token(qwen3_30b_a3b_bundle):
    """关键 MoE 收益断言: 单 token (decode 边界) 读字节 / 总存储 ≈ top_k / num_experts。

    这是 MoE 在 decode 单 token 时的"激活率"上界。对 tokens>1 的情形, 比例由
    coupon collector 决定 (见 test_routed_experts_weight_read_scales_with_distinct)。
    """
    b = qwen3_30b_a3b_bundle
    m, deploy = b.model, b.deploy

    expert_dim_per_device = m.expert_dim // deploy.tp
    layer_storage_bytes = (
        m.num_experts * 3 * m.hidden_dim * expert_dim_per_device * deploy.w_byte
    )
    per_token_read = (
        m.num_activated_experts * 3 * m.hidden_dim * expert_dim_per_device * deploy.w_byte
    )
    ratio = per_token_read / layer_storage_bytes
    expected_ratio = m.num_activated_experts / m.num_experts  # 8/128 = 0.0625
    assert ratio == pytest.approx(expected_ratio)


def test_skew_one_pins_routed_experts_back_to_topk(qwen3_30b_a3b_bundle):
    """skew=1 (极端 imbalance) 时, weight_read 退化到 top_k 公式 (worst-case)。

    用 backend.moe_routing.skew=1 重新构造 cost path, 验证 distinct=top_k。
    """
    from copy import deepcopy
    b = deepcopy(qwen3_30b_a3b_bundle)
    b.backend.moe_routing.skew = 1.0

    from llm_infer_sim.core.cost_model.model_core import ModelCoreCostModel
    from llm_infer_sim.core.workload.workload import (
        GlobalStepWorkload, RequestWorkload, StepPhase,
    )

    mc = ModelCoreCostModel(b)
    wl = GlobalStepWorkload(
        step_id=1, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=128, context_len=128, target_output_len=8,
        )],
        num_prefill_tokens=128, num_decode_tokens=0,
        total_scheduled_tokens=128,
        num_prefill_requests=1, num_decode_requests=0,
    )
    # Find a moe layer's routed_experts mem_bytes in per_op breakdown
    result = mc.estimate(wl)
    moe_routed_ops = [
        op for op in result["per_op"]
        if op["name"] == "routed_experts" and "layer" in op["scope"]
    ]
    assert moe_routed_ops, "no routed_experts op found in result['per_op']"

    # skew=1 → weight_read = top_k × 3 × h × expert_dim/tp × w_byte
    m, deploy = b.model, b.deploy
    expert_dim_per_device = m.expert_dim // deploy.tp
    expected_weight = (
        m.num_activated_experts * 3 * m.hidden_dim * expert_dim_per_device * deploy.w_byte
    )
    expected_act = 2 * 128 * m.hidden_dim * deploy.a_byte  # tokens=128 Q+O
    expected_mem = int(expected_weight + expected_act)
    actual_mem = moe_routed_ops[0]["mem_bytes"]
    assert actual_mem == expected_mem, (
        f"skew=1 should pin weight_read back to top_k formula. "
        f"Got mem_bytes={actual_mem}, expected {expected_mem}"
    )
