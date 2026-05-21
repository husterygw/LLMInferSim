"""Normalization and elementwise Operator factory."""
from __future__ import annotations

from llm_infer_sim.core.operators.factories.common import dense_parallel, make_runtime
from llm_infer_sim.core.operators.ops import ElementwiseOp, NormOp
from llm_infer_sim.core.operators.specs import OperatorFormula
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

    def _norm(self, subtype: str, layer_idx: int, tokens: int, phase: str) -> NormOp:
        elements = tokens * self.model.hidden_dim
        return NormOp(
            name=f"layer{layer_idx}_{subtype}",
            op_kind="norm",
            op_subtype=subtype,
            phase=phase,
            layer_idx=layer_idx,
            dtype="bf16",
            shape_fields={"tokens": tokens, "hidden": self.model.hidden_dim},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="norm",
                flops=elements * 4,
                load_act=int(elements * self.a_byte),
                store_act=int(elements * self.a_byte),
            ),
        )

    def _residual(
        self, subtype: str, layer_idx: int, tokens: int, phase: str,
    ) -> ElementwiseOp:
        elements = tokens * self.model.hidden_dim
        return ElementwiseOp(
            name=f"layer{layer_idx}_{subtype}",
            op_kind="elementwise",
            op_subtype=subtype,
            phase=phase,
            layer_idx=layer_idx,
            dtype="bf16",
            shape_fields={"tokens": tokens, "hidden": self.model.hidden_dim},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="activation",
                flops=elements,
                load_act=int(elements * self.a_byte),
                store_act=int(elements * self.a_byte),
            ),
        )

    def attn_norm(self, layer_idx: int, tokens: int, phase: str) -> NormOp:
        return self._norm("attn_norm", layer_idx, tokens, phase)

    def mlp_norm(self, layer_idx: int, tokens: int, phase: str) -> NormOp:
        return self._norm("mlp_norm", layer_idx, tokens, phase)

    def attn_add(self, layer_idx: int, tokens: int, phase: str) -> ElementwiseOp:
        return self._residual("attn_add", layer_idx, tokens, phase)

    def mlp_add(self, layer_idx: int, tokens: int, phase: str) -> ElementwiseOp:
        return self._residual("mlp_add", layer_idx, tokens, phase)

    # ---- V4 Hyper-Connection pre/post (hc_mult > 0) ----

    def hc_pre(self, layer_idx: int, tokens: int, phase: str,
               *, scope: str, w_byte: float = 2.0) -> NormOp:
        """HC pre: RMSNorm + linear mix + Sinkhorn + weighted sum. scope = 'attn' / 'ffn'."""
        m = self.model
        hc_mult = m.hc_mult
        h = m.hidden_dim
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * h

        norm_flops = tokens * hc_dim * 7
        linear_flops = tokens * hc_dim * mix_hc * 2
        sinkhorn_flops = tokens * m.hc_sinkhorn_iters * hc_mult * hc_mult * 5
        wsum_flops = tokens * hc_mult * h
        total_flops = norm_flops + linear_flops + sinkhorn_flops + wsum_flops

        weight_io = int(mix_hc * hc_dim * 4 + mix_hc * 4 + 3 * 4)  # fp32 weights + biases
        load_act = int(tokens * hc_mult * h * self.a_byte)
        store_act = int(
            tokens * h * self.a_byte
            + tokens * hc_mult * 4
            + tokens * hc_mult * hc_mult * 4
        )

        return NormOp(
            name=f"hc_{scope}_pre", op_kind="norm", op_subtype=f"hc_{scope}_pre",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"tokens": tokens, "hidden": h, "hc_mult": hc_mult},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="norm",
                flops=total_flops,
                load_weight=weight_io,
                load_act=load_act,
                store_act=store_act,
            ),
        )

    def hc_post(self, layer_idx: int, tokens: int, phase: str,
                *, scope: str) -> ElementwiseOp:
        """HC post: expand 1 output → hc_mult copies (broadcast multiply + comb @ residual)."""
        m = self.model
        hc_mult = m.hc_mult
        h = m.hidden_dim
        broadcast_flops = tokens * hc_mult * h
        comb_flops = tokens * hc_mult * hc_mult * h
        total_flops = broadcast_flops + comb_flops
        load_act = int(tokens * h * self.a_byte + tokens * hc_mult * h * self.a_byte)
        store_act = int(tokens * hc_mult * h * self.a_byte)
        return ElementwiseOp(
            name=f"hc_{scope}_post", op_kind="elementwise", op_subtype=f"hc_{scope}_post",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"tokens": tokens, "hidden": h, "hc_mult": hc_mult},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="activation",
                flops=total_flops,
                load_act=load_act,
                store_act=store_act,
            ),
        )

    def mlp_act(self, layer_idx: int, tokens: int, phase: str) -> ElementwiseOp:
        inter_per_tp = self.model.ffn_dim // self.deploy.tp_size
        elements = tokens * inter_per_tp
        return ElementwiseOp(
            name=f"layer{layer_idx}_mlp_act",
            op_kind="elementwise",
            op_subtype="mlp_act",
            phase=phase,
            layer_idx=layer_idx,
            dtype="bf16",
            shape_fields={"tokens": tokens, "intermediate": inter_per_tp},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="activation",
                flops=elements * 5,
                load_act=int(elements * self.a_byte * 2),
                store_act=int(elements * self.a_byte),
            ),
        )
