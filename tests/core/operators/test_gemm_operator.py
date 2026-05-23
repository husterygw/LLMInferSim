from __future__ import annotations

from llm_infer_sim.core.operator_schema.gemm import gemm_case_params_to_signature
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.operators import GEMM
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig


def _ctx(*, tp=1, execution_mode="eager", framework_version="0.20.1"):
    return build_operator_context(
        ModelConfig(),
        DeployConfig(
            tp_size=tp,
            execution_mode=execution_mode,
            backend="vllm",
            backend_version=framework_version,
        ),
        get_hardware_profile("RTX_4090"),
    )


def test_gemm_op_exposes_signature_and_roofline_spec():
    op = GEMM(
        name="qkv_proj",
        op_subtype="qkv_proj",
        phase="prefill",
        layer_idx=0,
        m=128,
        n=6144,
        k=2560,
        ctx=_ctx(),
    )

    assert op.op_kind == "gemm"
    assert op.shape == {"m": 128, "n": 6144, "k": 2560}
    assert op.parallel == {"tp": 1}
    assert op.runtime["execution_mode"] == "eager"

    formula = op.roofline_spec()
    assert formula.flops == 2 * 128 * 6144 * 2560
    assert formula.load_weight == 6144 * 2560 * 2
    assert formula.load_act == 128 * 2560 * 2
    assert formula.store_act == 128 * 6144 * 2

    expected = gemm_case_params_to_signature(
        {
            "op_subtype": "qkv_proj",
            "m": 128,
            "n": 6144,
            "k": 2560,
            "dtype": "bf16",
            "tp": 1,
            "execution_mode": "eager",
        },
        framework="vllm",
        framework_version="0.20.1",
        kernel_source="vllm_default",
    )
    assert op.signature() == expected


def test_gemm_op_formula_dict_for_trace_payloads():
    op = GEMM(
        name="o_proj",
        op_subtype="o_proj",
        phase="decode",
        layer_idx=0,
        m=8,
        n=2560,
        k=4096,
        ctx=_ctx(execution_mode="cudagraph"),
        kernel_source="vllm_row_parallel_linear",
    )

    payload = op.roofline_spec().to_dict()
    assert payload["flops"] == op.roofline_spec().flops
    assert payload["op_category"] == "matmul"
