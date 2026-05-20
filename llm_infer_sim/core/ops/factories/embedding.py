"""EmbeddingOpFactory — token embedding lookup. 公式从 core/ops/embedding.py."""
from __future__ import annotations

from llm_infer_sim.core.graph.virtual_op import VirtualOp
from llm_infer_sim.core.ops.embedding import embedding as _embedding
from llm_infer_sim.core.ops.factories._common import (
    dense_parallel,
    make_runtime,
    profile_to_formula,
)
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

    def embedding(self, tokens: int, phase: str) -> VirtualOp:
        prof = _embedding(
            tokens=tokens,
            vocab_size=self.model.vocab_size,
            hidden_dim=self.model.hidden_dim,
            w_byte=self.w_byte, a_byte=self.a_byte,
        )
        return VirtualOp(
            name="embedding",
            op_kind="embedding", op_subtype="embedding",
            phase=phase, layer_idx=None, dtype="bf16",
            shape={
                "tokens": tokens,
                "vocab_size": self.model.vocab_size,
                "hidden": self.model.hidden_dim,
            },
            parallel=dense_parallel(self.deploy),
            runtime=make_runtime(self.deploy),
            formula=profile_to_formula(prof),
        )
