"""IndexerOpFactory — V3.2 DSA lightning indexer + V4 indexer.

V3.2 indexer 块 (5 ops):
    indexer_wq_b           — q_lora_rank → n_head × head_dim (ReplicatedLinear, fp8/fp16)
    indexer_wk_weights_proj — h → (head_dim + n_head) fused (bf16 unquant)
    indexer_k_norm          — LayerNorm on head_dim
    indexer_q_fp8_quant     — per-token group fp8 quant on Q
    sparse_attn_indexer     — Q×K_cache → top-k indices (custom kernel)

V4 indexer 块 (compress_ratio==4 时):
    fused_index_compress_wkv_wgate
    index_wq_b              — q_lora_rank → index_n_heads × index_head_dim (no /tp!)
    index_weights_proj      — h → index_n_heads (no /tp!)
    index_score             — Q_index × K_compressed → scores → topk
"""
from __future__ import annotations

from llm_infer_sim.core.operators.factories.common import dense_parallel, make_runtime
from llm_infer_sim.core.operators.ops import ElementwiseOp, FormulaOp, NormOp
from llm_infer_sim.core.operators.specs import OperatorFormula
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig


class IndexerOpFactory:
    def __init__(
        self,
        model: ModelConfig,
        deploy: DeployConfig,
        *,
        w_byte: float = 2.0,
        a_byte: float = 2.0,
        kv_byte: float = 2.0,
        indexer_kv_byte: float = 1.0,
    ):
        self.model = model
        self.deploy = deploy
        self.w_byte = w_byte
        self.a_byte = a_byte
        self.kv_byte = kv_byte
        self.indexer_kv_byte = indexer_kv_byte

    # ---------- V3.2 lightning indexer ----------

    def v32_indexer_block(self, layer_idx: int, tokens: int, phase: str,
                          ctx_len: int) -> list:
        """5 ops + sparse_attn_indexer 共 5 个 (跟旧 _build_v32_indexer_ops 一致)."""
        m = self.model
        h = m.hidden_dim
        n_head = m.index_n_heads
        head_dim = m.index_head_dim
        op_prec_idx = "fp8" if self.w_byte <= 1.0 else "fp16"
        prefix = f"layer{layer_idx}"

        ops: list = []

        # indexer_wq_b: q_lora_rank → n_head × head_dim, ReplicatedLinear (no /tp), fp8.
        ops.append(self._linear_op(
            name=f"{prefix}_indexer_wq_b",
            op_subtype="indexer_wq_b",
            layer_idx=layer_idx, phase=phase,
            ic=m.q_lora_rank, oc=n_head * head_dim,
            tokens=tokens, op_precision=op_prec_idx,
        ))

        # indexer_wk_weights_proj: hidden → (head_dim + n_head), bf16 unquant.
        fused_oc = head_dim + n_head
        ops.append(FormulaOp(
            name=f"{prefix}_indexer_wk_weights_proj",
            op_kind="gemm", op_subtype="indexer_wk_weights_proj",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"m": tokens, "n": fused_oc, "k": h},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_replicated_linear"),
            formula_value=OperatorFormula(
                op_category="matmul",
                flops=h * fused_oc * tokens * 2,
                load_weight=int(h * fused_oc * 2.0),
                load_act=int(h * tokens * self.a_byte),
                store_act=int(fused_oc * tokens * 2.0),
                op_precision="bf16",
            ),
        ))

        # indexer_k_norm: LayerNorm on head_dim
        ops.append(NormOp(
            name=f"{prefix}_indexer_k_norm",
            op_kind="norm", op_subtype="rmsnorm",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"tokens": tokens, "hidden": head_dim},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="norm",
                flops=tokens * head_dim * 4,
                load_act=int(tokens * head_dim * self.a_byte),
                store_act=int(tokens * head_dim * self.a_byte),
            ),
        ))

        # indexer_q_fp8_quant: per-token group fp8 quant on Q
        q_size = tokens * n_head * head_dim
        ops.append(ElementwiseOp(
            name=f"{prefix}_indexer_q_fp8_quant",
            op_kind="elementwise", op_subtype="quantize",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"tokens": tokens, "n_head": n_head, "head_dim": head_dim},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy),
            formula_value=OperatorFormula(
                op_category="activation",
                flops=q_size * 5,
                load_act=int(q_size * self.a_byte),
                store_act=int(q_size * 1.0 + q_size // 128 * 4),
            ),
        ))

        # sparse_attn_indexer: Q×K_cache → top-k
        scale_bytes_per_pos = 4 * (head_dim // 128)
        ops.append(FormulaOp(
            name=f"{prefix}_sparse_attn_indexer",
            op_kind="attention", op_subtype="sparse_index",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            tags=("v32_indexer",),
            shape_fields={
                "tokens": tokens, "ctx_len": ctx_len,
                "n_head": n_head, "head_dim": head_dim,
                "index_topk": m.index_topk,
            },
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_sparse_attn_indexer"),
            formula_value=OperatorFormula(
                op_category="attention",
                flops=tokens * ctx_len * head_dim * n_head * 2,
                load_act=int(tokens * n_head * head_dim * 1.0),
                load_kv_cache=int(ctx_len * (head_dim * self.indexer_kv_byte + scale_bytes_per_pos)),
                store_act=int(tokens * m.index_topk * 4),
            ),
        ))
        return ops

    # ---------- V4 indexer (compress_ratio == 4 only) ----------

    def v4_indexer_ops(self, *, layer_idx: int, tokens: int, ctx_len: int,
                        phase: str, compress_ratio: int, op_prec: str) -> list:
        """V4 indexer 3 ops (fused_index_compress_wkv_wgate + index_wq_b + index_weights_proj + index_score).

        ReplicatedLinear (no /tp): index_wq_b 和 index_weights_proj 每 rank 跑完整 index_n_heads.
        """
        m = self.model
        h = m.hidden_dim
        prefix = f"layer{layer_idx}"

        ops: list = []
        idx_compress_out_dim = 2 * m.index_head_dim
        idx_fused_oc = 2 * idx_compress_out_dim
        # fused_index_compress_wkv_wgate (bf16)
        ops.append(FormulaOp(
            name=f"{prefix}_fused_index_compress_wkv_wgate",
            op_kind="gemm", op_subtype="fused_index_compress_wkv_wgate",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"m": tokens, "n": idx_fused_oc, "k": h},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_fused_index_compress"),
            formula_value=OperatorFormula(
                op_category="matmul",
                flops=h * idx_fused_oc * tokens * 2,
                load_weight=int(h * idx_fused_oc * 2.0),
                load_act=int(h * tokens * self.a_byte),
                store_act=int(idx_fused_oc * tokens * 2.0),
                op_precision="bf16",
            ),
        ))

        # index_wq_b: q_lora_rank → index_n_heads × index_head_dim (ReplicatedLinear, no /tp)
        index_full_heads = m.index_n_heads
        ops.append(self._linear_op(
            name=f"{prefix}_index_wq_b",
            op_subtype="index_wq_b",
            layer_idx=layer_idx, phase=phase,
            ic=m.q_lora_rank, oc=index_full_heads * m.index_head_dim,
            tokens=tokens, op_precision=op_prec,
        ))

        # index_weights_proj: h → index_n_heads (ReplicatedLinear, no /tp, bf16)
        ops.append(self._linear_op(
            name=f"{prefix}_index_weights_proj",
            op_subtype="index_weights_proj",
            layer_idx=layer_idx, phase=phase,
            ic=h, oc=index_full_heads,
            tokens=tokens,
            op_precision="bf16",
            override_w_byte=2.0,
        ))

        # index_score: Q_index × K_compressed → top-k scores
        cache_compressed_size = ctx_len // compress_ratio if compress_ratio > 0 else 0
        ops.append(FormulaOp(
            name=f"{prefix}_index_score",
            op_kind="attention", op_subtype="index_score",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            tags=("v4_indexer",),
            shape_fields={
                "tokens": tokens, "cache_compressed_size": cache_compressed_size,
                "index_n_heads": index_full_heads,
                "index_head_dim": m.index_head_dim,
            },
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_index_score"),
            formula_value=OperatorFormula(
                op_category="attention",
                flops=tokens * cache_compressed_size * m.index_head_dim * index_full_heads * 2,
                load_act=int(tokens * index_full_heads * m.index_head_dim * self.a_byte),
                load_kv_cache=int(cache_compressed_size * m.index_head_dim
                                  * tokens * self.indexer_kv_byte),
                store_act=int(tokens * cache_compressed_size * index_full_heads * self.a_byte),
            ),
        ))
        return ops

    # ---------- helper ----------

    def _linear_op(self, *, name: str, op_subtype: str, layer_idx: int,
                    phase: str, ic: int, oc: int, tokens: int,
                    op_precision: str, override_w_byte: float | None = None):
        w_byte = override_w_byte if override_w_byte is not None else self.w_byte
        return FormulaOp(
            name=name, op_kind="gemm", op_subtype=op_subtype,
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"m": tokens, "n": oc, "k": ic},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_replicated_linear"),
            formula_value=OperatorFormula(
                op_category="matmul",
                flops=2 * tokens * ic * oc,
                load_weight=int(ic * oc * w_byte),
                load_act=int(tokens * ic * self.a_byte),
                store_act=int(tokens * oc * self.a_byte),
                op_precision=op_precision,
            ),
        )
