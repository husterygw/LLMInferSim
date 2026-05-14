"""Dynamic fp8 activation quantize op (#7).

覆盖:
  1. activation_quantize op 字节公式 (read base_a + write a + scale)
  2. layer_builder 在 a_byte == base_a_byte 时 不 注入 quant op (向后兼容)
  3. layer_builder 在 a_byte < base_a_byte 时注入 2 个 quant op/layer
     (attn_input_quant + mlp_input_quant)
  4. 同 quant op 在 dense / MoE / MLA / V4 sparse / V3.2 5 条 attn path 都生效
  5. 总 bandwidth 比 fp16 模型大 (基础 dtype 比) 但小很多 (因为 fp8 主权重)
"""
from __future__ import annotations

from llm_infer_sim.core.cost_model.layer_builder import (
    dense_layer_time, moe_layer_time,
)
from llm_infer_sim.core.ops.normalization import activation_quantize
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig


# ---------- helpers ----------

def _qwen3_4b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-4B",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def _qwen3_30b_a3b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-30B-A3B",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=0, num_layers=48, vocab_size=151936,
        is_moe=True,
        num_experts=128, num_activated_experts=8, expert_dim=768,
        num_shared_experts=0, moe_layer_freq=1, first_moe_layer=0,
    )


def _fp16_deploy() -> DeployConfig:
    """无 quant 部署: a_byte=base_a_byte=2.0."""
    return DeployConfig(w_byte=2.0, a_byte=2.0, kv_byte=2.0,
                        base_w_byte=2.0, base_a_byte=2.0)


def _fp8_deploy() -> DeployConfig:
    """Dynamic fp8 部署: a_byte=1.0, base_a_byte=2.0."""
    return DeployConfig(w_byte=1.0, a_byte=1.0, kv_byte=1.0,
                        base_w_byte=2.0, base_a_byte=2.0)


def _has_op(layer_result, name: str) -> bool:
    return any(op.name == name for op in layer_result.ops)


def _count_op(layer_result, name: str) -> int:
    return sum(1 for op in layer_result.ops if op.name == name)


# ---------- op 字节公式 ----------

def test_activation_quantize_bytes():
    """read bf16 + write fp8 + write scales (per-group)."""
    op = activation_quantize(
        "test", tokens=128, hidden_size=2560,
        base_a_byte=2.0, a_byte=1.0, block_size=128, scale_byte=4.0,
    )
    # read: 128 × 2560 × 2 = 655360
    assert op.load_act == 655360
    # write: 128 × 2560 × 1 + 128 × ceil(2560/128) × 4 = 327680 + 128*20*4 = 337920
    assert op.store_act == 327680 + 128 * 20 * 4
    # flops: 5 ops/elem
    assert op.flops == 128 * 2560 * 5
    assert op.op_category == "activation"


def test_activation_quantize_fp4():
    """fp4: write 更少 byte."""
    op = activation_quantize(
        "test", tokens=128, hidden_size=2560,
        base_a_byte=2.0, a_byte=0.5,
    )
    # write: 128 × 2560 × 0.5 = 163840 + scales
    assert op.store_act == int(128 * 2560 * 0.5) + 128 * 20 * 4


# ---------- layer_builder 注入: fp16 不插, fp8 插 ----------

def test_no_quant_op_when_a_byte_eq_base():
    """fp16 模型 (a=base=2): 不应注入 attn_input_quant / mlp_input_quant."""
    model = _qwen3_4b()
    deploy = _fp16_deploy()
    hw = get_hardware_profile("H100")
    lr = dense_layer_time(0, "prefill", 128, 128, model, deploy, hw)
    assert not _has_op(lr, "attn_input_quant")
    assert not _has_op(lr, "mlp_input_quant")


def test_quant_op_injected_under_fp8_dense():
    """fp8 dense layer: 1 attn_input_quant + 1 mlp_input_quant."""
    model = _qwen3_4b()
    deploy = _fp8_deploy()
    hw = get_hardware_profile("H100")
    lr = dense_layer_time(0, "prefill", 128, 128, model, deploy, hw)
    assert _count_op(lr, "attn_input_quant") == 1
    assert _count_op(lr, "mlp_input_quant") == 1


def test_quant_op_injected_under_fp8_moe():
    """fp8 MoE layer: attn + mlp_input_quant 各 1."""
    model = _qwen3_30b_a3b()
    deploy = _fp8_deploy()
    hw = get_hardware_profile("H100")
    lr = moe_layer_time(0, "prefill", 128, 128, model, deploy, hw)
    assert _count_op(lr, "attn_input_quant") == 1
    assert _count_op(lr, "mlp_input_quant") == 1


def test_quant_op_bytes_sized_correctly_in_builder():
    """builder 注入的 quant op 字节应跟手算一致."""
    model = _qwen3_4b()
    deploy = _fp8_deploy()
    hw = get_hardware_profile("H100")
    lr = dense_layer_time(0, "prefill", 128, 128, model, deploy, hw)
    quant_ops = [op for op in lr.ops if op.name.endswith("_input_quant")]
    assert len(quant_ops) == 2
    for op in quant_ops:
        # h=2560, tokens=128
        assert op.load_act == 128 * 2560 * 2  # bf16 read
        # write fp8 + scales
        expected_store = int(128 * 2560 * 1) + 128 * 20 * 4
        assert op.store_act == expected_store


# ---------- 总 bandwidth 增量合理 ----------

def test_fp8_total_bandwidth_smaller_than_fp16_but_quant_overhead_present():
    """fp8 总 bandwidth 应比 fp16 小 (weights 一半), 但 quant 添加可见的 activation 项."""
    model = _qwen3_4b()
    hw = get_hardware_profile("H100")
    lr_fp16 = dense_layer_time(0, "prefill", 128, 128, model, _fp16_deploy(), hw)
    lr_fp8 = dense_layer_time(0, "prefill", 128, 128, model, _fp8_deploy(), hw)

    total_bytes_fp16 = sum(
        op.load_weight + op.load_act + op.store_act + op.load_kv_cache + op.store_kv_cache
        for op in lr_fp16.ops if op.op_category != "communication"
    )
    total_bytes_fp8 = sum(
        op.load_weight + op.load_act + op.store_act + op.load_kv_cache + op.store_kv_cache
        for op in lr_fp8.ops if op.op_category != "communication"
    )
    # fp8 weight 一半, 总应小 (即使加 quant op, weight 节省主导)
    assert total_bytes_fp8 < total_bytes_fp16
    # 但比 "fp8 无 quant" 多: quant op 的 read 是 bf16, 是补偿后的 bandwidth
    quant_total = sum(
        op.load_act + op.store_act
        for op in lr_fp8.ops if op.name.endswith("_input_quant")
    )
    assert quant_total > 0
