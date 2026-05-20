"""DenseOpFactory — V3 §6.5 + IMPL_PLAN §1.4 Step 1.7.

生成 dense GEMM 类 VirtualOp (qkv_proj / o_proj / gate_up_proj / down_proj / lm_head).
公式从 core/ops/linear.py + core/ops/embedding.py 迁移.

阶段 1 范围:
  - BF16 unquantized (w_byte=a_byte=kv_byte=2)
  - TP 维度通过 deploy.tp_size 控制 (per-rank 切分)
"""
from __future__ import annotations

from llm_infer_sim.core.graph.virtual_op import VirtualOp
from llm_infer_sim.core.ops.embedding import lm_head as _lm_head
from llm_infer_sim.core.ops.factories._common import (
    dense_parallel,
    make_runtime,
    profile_to_formula,
)
from llm_infer_sim.core.ops.linear import (
    fused_gate_up_gemm,
    fused_qkv_gemm,
    linear_layer,
)
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig


class DenseOpFactory:
    def __init__(
        self,
        model: ModelConfig,
        deploy: DeployConfig,
        *,
        w_byte: float = 2.0,
        a_byte: float = 2.0,
        kv_byte: float = 2.0,
    ):
        self.model = model
        self.deploy = deploy
        self.w_byte = w_byte
        self.a_byte = a_byte
        self.kv_byte = kv_byte

    # ---- per-layer GEMMs ----

    def qkv_proj(self, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        tp = self.deploy.tp_size
        n_q = self.model.num_heads // tp
        n_kv = self.model.num_kv_heads // tp
        prof = fused_qkv_gemm(
            name="qkv_proj",
            hidden=self.model.hidden_dim,
            num_q_heads_per_tp=n_q,
            num_kv_heads_per_tp=n_kv,
            head_dim=self.model.head_dim,
            tokens=tokens,
            w_byte=self.w_byte, a_byte=self.a_byte, kv_byte=self.kv_byte,
        )
        n_total = (n_q + 2 * n_kv) * self.model.head_dim
        return VirtualOp(
            name=f"layer{layer_idx}_qkv_proj",
            op_kind="gemm", op_subtype="qkv_proj",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape={"m": tokens, "n": n_total, "k": self.model.hidden_dim},
            parallel=dense_parallel(self.deploy),
            runtime=make_runtime(self.deploy),
            formula=profile_to_formula(prof),
        )

    def o_proj(self, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        tp = self.deploy.tp_size
        n_q = self.model.num_heads // tp
        ic = n_q * self.model.head_dim
        oc = self.model.hidden_dim
        prof = linear_layer(
            name="o_proj", ic=ic, oc=oc, tokens=tokens,
            w_byte=self.w_byte, a_byte=self.a_byte, kv_byte=self.kv_byte,
        )
        return VirtualOp(
            name=f"layer{layer_idx}_o_proj",
            op_kind="gemm", op_subtype="o_proj",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape={"m": tokens, "n": oc, "k": ic},
            parallel=dense_parallel(self.deploy),
            runtime=make_runtime(self.deploy),
            formula=profile_to_formula(prof),
        )

    def gate_up_proj(self, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        tp = self.deploy.tp_size
        inter_per_tp = self.model.ffn_dim // tp
        prof = fused_gate_up_gemm(
            name="gate_up_proj",
            hidden=self.model.hidden_dim,
            intermediate_per_tp=inter_per_tp,
            tokens=tokens,
            w_byte=self.w_byte, a_byte=self.a_byte,
        )
        return VirtualOp(
            name=f"layer{layer_idx}_gate_up_proj",
            op_kind="gemm", op_subtype="gate_up_proj",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape={"m": tokens, "n": 2 * inter_per_tp, "k": self.model.hidden_dim},
            parallel=dense_parallel(self.deploy),
            runtime=make_runtime(self.deploy),
            formula=profile_to_formula(prof),
        )

    def down_proj(self, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        tp = self.deploy.tp_size
        inter_per_tp = self.model.ffn_dim // tp
        oc = self.model.hidden_dim
        prof = linear_layer(
            name="down_proj", ic=inter_per_tp, oc=oc, tokens=tokens,
            w_byte=self.w_byte, a_byte=self.a_byte, kv_byte=self.kv_byte,
        )
        return VirtualOp(
            name=f"layer{layer_idx}_down_proj",
            op_kind="gemm", op_subtype="down_proj",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape={"m": tokens, "n": oc, "k": inter_per_tp},
            parallel=dense_parallel(self.deploy),
            runtime=make_runtime(self.deploy),
            formula=profile_to_formula(prof),
        )

    # ---- model-level GEMM (V3 §6.5 把 lm_head 归 DenseOpFactory) ----

    def lm_head(self, tokens: int, phase: str) -> VirtualOp:
        tp = self.deploy.tp_size
        oc = self.model.vocab_size // tp
        prof = _lm_head(
            tokens=tokens,
            vocab_size=self.model.vocab_size,
            hidden_dim=self.model.hidden_dim,
            tp_size=tp,
            w_byte=self.w_byte, a_byte=self.a_byte,
        )
        return VirtualOp(
            name="lm_head",
            op_kind="gemm", op_subtype="lm_head",
            phase=phase, layer_idx=None, dtype="bf16",
            shape={"m": tokens, "n": oc, "k": self.model.hidden_dim},
            parallel=dense_parallel(self.deploy),
            runtime=make_runtime(self.deploy),
            formula=profile_to_formula(prof),
        )
