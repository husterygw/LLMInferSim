"""Dense GEMM Operator factory."""
from __future__ import annotations

from llm_infer_sim.core.operators.factories.common import make_runtime
from llm_infer_sim.core.operators.ops import GemmOp
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

    def linear(
        self,
        *,
        name: str,
        op_subtype: str,
        layer_idx: int | None,
        tokens: int,
        phase: str,
        ic: int,
        oc: int,
        is_kv_proj: bool = False,
        kernel_source: str = "vllm_row_parallel_linear",
    ) -> GemmOp:
        """Generic linear (h_in -> h_out) for non-fused projections.

        is_kv_proj=True: 输出走 store_kv_cache, byte 用 self.kv_byte (e.g. MLA kv_a_proj_with_mqa).
        否则输出走 store_act, byte 用 self.a_byte.
        """
        out_byte = self.kv_byte if is_kv_proj else self.a_byte
        runtime = make_runtime(self.deploy, kernel_source=kernel_source)
        return GemmOp(
            name=name,
            op_subtype=op_subtype,
            phase=phase,
            layer_idx=layer_idx,
            dtype="bf16",
            m=tokens,
            n=oc,
            k=ic,
            tp=self.deploy.tp_size,
            framework=runtime["framework"],
            framework_version=runtime["framework_version"],
            execution_mode=runtime["execution_mode"],
            kernel_source=runtime["kernel_source"],
            weight_bytes_per_elem=self.w_byte,
            act_bytes_per_elem=self.a_byte,
            out_bytes_per_elem=out_byte,
            is_kv_proj=is_kv_proj,
        )

    def _gemm(
        self,
        *,
        name: str,
        op_subtype: str,
        phase: str,
        layer_idx: int | None,
        m: int,
        n: int,
        k: int,
    ) -> GemmOp:
        runtime = make_runtime(self.deploy, kernel_source="vllm_row_parallel_linear")
        return GemmOp(
            name=name,
            op_subtype=op_subtype,
            phase=phase,
            layer_idx=layer_idx,
            dtype="bf16",
            m=m,
            n=n,
            k=k,
            tp=self.deploy.tp_size,
            framework=runtime["framework"],
            framework_version=runtime["framework_version"],
            execution_mode=runtime["execution_mode"],
            kernel_source=runtime["kernel_source"],
            weight_bytes_per_elem=self.w_byte,
            act_bytes_per_elem=self.a_byte,
            out_bytes_per_elem=self.kv_byte,
        )

    def qkv_proj(self, layer_idx: int, tokens: int, phase: str) -> GemmOp:
        tp = self.deploy.tp_size
        n_q = self.model.num_heads // tp
        n_kv = self.model.num_kv_heads // tp
        n_total = (n_q + 2 * n_kv) * self.model.head_dim
        return self._gemm(
            name=f"layer{layer_idx}_qkv_proj",
            op_subtype="qkv_proj",
            phase=phase,
            layer_idx=layer_idx,
            m=tokens,
            n=n_total,
            k=self.model.hidden_dim,
        )

    def o_proj(self, layer_idx: int, tokens: int, phase: str) -> GemmOp:
        tp = self.deploy.tp_size
        n_q = self.model.num_heads // tp
        return self._gemm(
            name=f"layer{layer_idx}_o_proj",
            op_subtype="o_proj",
            phase=phase,
            layer_idx=layer_idx,
            m=tokens,
            n=self.model.hidden_dim,
            k=n_q * self.model.head_dim,
        )

    def gate_up_proj(self, layer_idx: int, tokens: int, phase: str) -> GemmOp:
        inter_per_tp = self.model.ffn_dim // self.deploy.tp_size
        return self._gemm(
            name=f"layer{layer_idx}_gate_up_proj",
            op_subtype="gate_up_proj",
            phase=phase,
            layer_idx=layer_idx,
            m=tokens,
            n=2 * inter_per_tp,
            k=self.model.hidden_dim,
        )

    def down_proj(self, layer_idx: int, tokens: int, phase: str) -> GemmOp:
        inter_per_tp = self.model.ffn_dim // self.deploy.tp_size
        return self._gemm(
            name=f"layer{layer_idx}_down_proj",
            op_subtype="down_proj",
            phase=phase,
            layer_idx=layer_idx,
            m=tokens,
            n=self.model.hidden_dim,
            k=inter_per_tp,
        )

    def lm_head(self, tokens: int, phase: str) -> GemmOp:
        return self._gemm(
            name="lm_head",
            op_subtype="lm_head",
            phase=phase,
            layer_idx=None,
            m=tokens,
            n=self.model.vocab_size // self.deploy.tp_size,
            k=self.model.hidden_dim,
        )
