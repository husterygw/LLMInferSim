"""NormalizationOpFactory — V3 §6.5 + IMPL_PLAN §1.4 Step 1.7.

attn_norm / mlp_norm / attn_add / mlp_add / mlp_act. 公式从 core/ops/normalization.py.
"""
from __future__ import annotations

from llm_infer_sim.core.graph.virtual_op import VirtualOp
from llm_infer_sim.core.ops.factories._common import (
    dense_parallel,
    make_runtime,
    profile_to_formula,
)
from llm_infer_sim.core.ops.normalization import (
    mlp_activation,
    norm_layer,
    residual_add,
)
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig


class NormalizationOpFactory:
    def __init__(
        self,
        model: ModelConfig,
        deploy: DeployConfig,
        *,
        a_byte: float = 2.0,
    ):
        self.model = model
        self.deploy = deploy
        self.a_byte = a_byte

    def _norm(self, subtype: str, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        prof = norm_layer(
            name=subtype,
            tokens=tokens,
            hidden_size=self.model.hidden_dim,
            a_byte=self.a_byte,
        )
        return VirtualOp(
            name=f"layer{layer_idx}_{subtype}",
            op_kind="norm", op_subtype=subtype,
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape={"tokens": tokens, "hidden": self.model.hidden_dim},
            parallel=dense_parallel(self.deploy),
            runtime=make_runtime(self.deploy),
            formula=profile_to_formula(prof),
        )

    def _residual(self, subtype: str, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        prof = residual_add(
            name=subtype, tokens=tokens,
            hidden_size=self.model.hidden_dim,
            a_byte=self.a_byte,
        )
        return VirtualOp(
            name=f"layer{layer_idx}_{subtype}",
            op_kind="elementwise", op_subtype=subtype,
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape={"tokens": tokens, "hidden": self.model.hidden_dim},
            parallel=dense_parallel(self.deploy),
            runtime=make_runtime(self.deploy),
            formula=profile_to_formula(prof),
        )

    def attn_norm(self, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        return self._norm("attn_norm", layer_idx, tokens, phase)

    def mlp_norm(self, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        return self._norm("mlp_norm", layer_idx, tokens, phase)

    def attn_add(self, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        return self._residual("attn_add", layer_idx, tokens, phase)

    def mlp_add(self, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        return self._residual("mlp_add", layer_idx, tokens, phase)

    def mlp_act(self, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        """SwiGLU: out = silu(gate) * up. tokens × intermediate_per_tp 个输出."""
        tp = self.deploy.tp_size
        inter_per_tp = self.model.ffn_dim // tp
        prof = mlp_activation(
            name="mlp_act", tokens=tokens,
            hidden_size=inter_per_tp,
            a_byte=self.a_byte,
        )
        return VirtualOp(
            name=f"layer{layer_idx}_mlp_act",
            op_kind="elementwise", op_subtype="mlp_act",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape={"tokens": tokens, "intermediate": inter_per_tp},
            parallel=dense_parallel(self.deploy),
            runtime=make_runtime(self.deploy),
            formula=profile_to_formula(prof),
        )
