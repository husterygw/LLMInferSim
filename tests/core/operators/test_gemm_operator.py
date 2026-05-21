from __future__ import annotations

from llm_infer_sim.core.operator_schema.gemm import gemm_case_params_to_signature
from llm_infer_sim.core.operators.ops import GemmOp


def test_gemm_op_exposes_signature_and_formula():
    op = GemmOp(
        name="layer0_qkv_proj",
        op_subtype="qkv_proj",
        phase="prefill",
        layer_idx=0,
        dtype="bf16",
        m=128,
        n=6144,
        k=2560,
        tp=1,
        framework="vllm",
        framework_version="0.20.1",
        execution_mode="eager",
        kernel_source="vllm_default",
    )

    assert op.op_kind == "gemm"
    assert op.shape == {"m": 128, "n": 6144, "k": 2560}
    assert op.parallel == {"tp": 1}
    assert op.runtime["execution_mode"] == "eager"

    formula = op.formula()
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
    op = GemmOp(
        name="layer0_o_proj",
        op_subtype="o_proj",
        phase="decode",
        layer_idx=0,
        dtype="bf16",
        m=8,
        n=2560,
        k=4096,
        tp=1,
        framework="vllm",
        framework_version="0.20.1",
        execution_mode="cudagraph",
        kernel_source="vllm_row_parallel_linear",
    )

    payload = op.formula().to_dict()
    assert payload["flops"] == op.formula().flops
    assert payload["op_category"] == "matmul"
