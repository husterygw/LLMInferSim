"""Embedding Operator factory."""
from __future__ import annotations

from llm_infer_sim.core.operators.factories.common import dense_parallel, make_runtime
from llm_infer_sim.core.operators.ops import ElementwiseOp, EmbeddingOp, FormulaOp, NormOp
from llm_infer_sim.core.operators.specs import OperatorFormula
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig


class EmbeddingOpFactory:
    def __init__(
        self,
        model: ModelConfig,
        deploy: DeployConfig,
        *,
        w_byte: float = 2.0,
        a_byte: float = 2.0,
    ):
        self.model = model
        self.deploy = deploy
        self.w_byte = w_byte
        self.a_byte = a_byte

    # ---- V4 HC model-level ops (hc_mult > 0) ----

    def hc_embedding_repeat(self, tokens: int, phase: str) -> ElementwiseOp:
        """V4 HC: embedding 后实化 hc_mult 副本 (tokens × hc_mult × h × a_byte)."""
        m = self.model
        return ElementwiseOp(
            name="hc_embedding_repeat",
            op_kind="elementwise", op_subtype="hc_embedding_repeat",
            phase=phase, layer_idx=None, dtype="bf16",
            shape_fields={"tokens": tokens, "hidden": m.hidden_dim, "hc_mult": m.hc_mult},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="activation",
                flops=0,
                load_act=int(tokens * m.hidden_dim * self.a_byte),
                store_act=int(tokens * m.hc_mult * m.hidden_dim * self.a_byte),
            ),
        )

    def hc_head(self, tokens: int, phase: str) -> FormulaOp:
        """V4 HC head fuse: Linear([tokens, hc_mult*h] → [tokens, h]) + RMSNorm."""
        m = self.model
        return FormulaOp(
            name="hc_head",
            op_kind="gemm", op_subtype="hc_head",
            phase=phase, layer_idx=None, dtype="bf16",
            shape_fields={"m": tokens, "n": m.hidden_dim, "k": m.hc_mult * m.hidden_dim},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_hc_head"),
            formula_value=OperatorFormula(
                op_category="matmul",
                flops=2 * tokens * (m.hc_mult * m.hidden_dim) * m.hidden_dim,
                load_weight=int((m.hc_mult * m.hidden_dim) * m.hidden_dim * self.w_byte),
                load_act=int(tokens * m.hc_mult * m.hidden_dim * self.a_byte),
                store_act=int(tokens * m.hidden_dim * self.a_byte),
            ),
        )

    def final_norm(self, tokens: int, phase: str) -> NormOp:
        """Final RMSNorm 在 lm_head 之前."""
        h = self.model.hidden_dim
        return NormOp(
            name="final_norm",
            op_kind="norm", op_subtype="rmsnorm",
            phase=phase, layer_idx=None, dtype="bf16",
            shape_fields={"tokens": tokens, "hidden": h},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="norm",
                flops=tokens * h * 4,
                load_act=int(tokens * h * self.a_byte),
                store_act=int(tokens * h * self.a_byte),
            ),
        )

    def embedding(self, tokens: int, phase: str) -> EmbeddingOp:
        return EmbeddingOp(
            name="embedding",
            op_kind="embedding",
            op_subtype="embedding",
            phase=phase,
            layer_idx=None,
            dtype="bf16",
            shape_fields={
                "tokens": tokens,
                "vocab_size": self.model.vocab_size,
                "hidden": self.model.hidden_dim,
            },
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="embedding",
                flops=0,
                load_weight=int(self.model.vocab_size * self.model.hidden_dim * self.w_byte),
                store_act=int(tokens * self.model.hidden_dim * self.a_byte),
            ),
        )
