"""V4 sparse attention factory — fused_wqa_wkv / compressor / sparse attention / wo_a/wo_b.

参考旧 cost_model.layer_builder._build_v4_sparse_attention_block + _build_v4_compressor_ops.

V4 attention block (per layer):
    [hc_attn_pre] → attn_norm
    fused_wqa_wkv → q_norm → wq_b → kv_norm
    [compress_ratio>0] fused_compress_wkv_wgate + compress_pool
    [compress_ratio==4]  fused_index_compress_wkv_wgate + index_wq_b + index_weights_proj + index_score
    fused_sparse_attention
    wo_a → wo_b → [tp>1 allreduce]
    [hc_attn_post] or attn_add

Sparse attention公式: 每 query 命中 (window + compressed + index_topk + attn_sink) 个 positions.
"""
from __future__ import annotations

import math

from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.operators.factories.common import dense_parallel, make_runtime
from llm_infer_sim.core.operators.ops import AttentionOp, ElementwiseOp, FormulaOp, NormOp
from llm_infer_sim.core.operators.specs import OperatorFormula
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig


class V4AttentionOpFactory:
    """V4-specific attention ops. V3/V3.2 走 AttentionOpFactory."""

    def __init__(
        self,
        model: ModelConfig,
        deploy: DeployConfig,
        hw: HardwareConfig,
        *,
        w_byte: float = 2.0,
        a_byte: float = 2.0,
        kv_byte: float = 2.0,
    ):
        self.model = model
        self.deploy = deploy
        self.hw = hw
        self.w_byte = w_byte
        self.a_byte = a_byte
        self.kv_byte = kv_byte

    # ---- attention proj fusions ----

    def fused_wqa_wkv(self, layer_idx: int, tokens: int, phase: str) -> FormulaOp:
        """MergedColumnParallelLinear (disable_tp=True): h → q_lora + head_dim concat."""
        m = self.model
        h = m.hidden_dim
        wqa_wkv_oc = m.q_lora_rank + m.head_dim
        op_prec = "fp8" if self.w_byte <= 1.0 else "fp16"
        return FormulaOp(
            name=f"layer{layer_idx}_fused_wqa_wkv",
            op_kind="gemm", op_subtype="fused_wqa_wkv",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"m": tokens, "n": wqa_wkv_oc, "k": h},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_fused_wqa_wkv"),
            formula_value=OperatorFormula(
                op_category="matmul",
                flops=h * wqa_wkv_oc * tokens * 2,
                load_weight=int(h * wqa_wkv_oc * self.w_byte),
                load_act=int(h * tokens * self.a_byte),       # x 一次
                store_act=int(wqa_wkv_oc * tokens * self.a_byte),
                op_precision=op_prec,
            ),
        )

    def q_norm(self, layer_idx: int, tokens: int, phase: str) -> NormOp:
        h = self.model.q_lora_rank
        return NormOp(
            name=f"layer{layer_idx}_q_norm", op_kind="norm", op_subtype="rmsnorm",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
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

    def wq_b(self, layer_idx: int, tokens: int, phase: str) -> FormulaOp:
        m = self.model
        tp = self.deploy.tp_size
        heads_per_tp = m.num_heads // tp
        op_prec = "fp8" if self.w_byte <= 1.0 else "fp16"
        return FormulaOp(
            name=f"layer{layer_idx}_wq_b",
            op_kind="gemm", op_subtype="wq_b",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"m": tokens, "n": heads_per_tp * m.head_dim, "k": m.q_lora_rank},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_row_parallel_linear"),
            formula_value=OperatorFormula(
                op_category="matmul",
                flops=2 * tokens * m.q_lora_rank * heads_per_tp * m.head_dim,
                load_weight=int(m.q_lora_rank * heads_per_tp * m.head_dim * self.w_byte),
                load_act=int(tokens * m.q_lora_rank * self.a_byte),
                store_act=int(tokens * heads_per_tp * m.head_dim * self.a_byte),
                op_precision=op_prec,
            ),
        )

    def kv_norm(self, layer_idx: int, tokens: int, phase: str) -> NormOp:
        h = self.model.head_dim
        return NormOp(
            name=f"layer{layer_idx}_kv_norm", op_kind="norm", op_subtype="rmsnorm",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
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

    # ---- compressor (fused) ----

    def compressor_ops(self, layer_idx: int, tokens: int, ctx_len: int,
                       phase: str, compress_ratio: int) -> list:
        """fused_compress_wkv_wgate (bf16) + compress_pool (输出到 KV cache)."""
        m = self.model
        h = m.hidden_dim
        coff = 2 if compress_ratio == 4 else 1
        compress_out_dim = coff * m.head_dim
        fused_oc = 2 * compress_out_dim

        compressed_tokens = tokens // compress_ratio if compress_ratio > 0 else 0

        ops: list = [
            FormulaOp(
                name=f"layer{layer_idx}_fused_compress_wkv_wgate",
                op_kind="gemm", op_subtype="fused_compress_wkv_wgate",
                phase=phase, layer_idx=layer_idx, dtype="bf16",
                shape_fields={"m": tokens, "n": fused_oc, "k": h},
                parallel_fields=dense_parallel(self.deploy),
                runtime_fields=make_runtime(self.deploy, kernel_source="vllm_fused_compress_wkv_wgate"),
                formula_value=OperatorFormula(
                    op_category="matmul",
                    flops=h * fused_oc * tokens * 2,
                    load_weight=int(h * fused_oc * 2.0),     # bf16 (quant_config=None)
                    load_act=int(h * tokens * self.a_byte),
                    store_act=int(fused_oc * tokens * 2.0),
                    op_precision="bf16",
                ),
            ),
            ElementwiseOp(
                name=f"layer{layer_idx}_compress_pool",
                op_kind="elementwise", op_subtype="compress_pool",
                phase=phase, layer_idx=layer_idx, dtype="bf16",
                shape_fields={"tokens": tokens, "out_dim": compress_out_dim,
                              "compressed_tokens": compressed_tokens},
                parallel_fields=dense_parallel(self.deploy),
                runtime_fields=make_runtime(self.deploy),
                formula_value=OperatorFormula(
                    op_category="activation",
                    flops=tokens * compress_out_dim * 5,
                    load_act=int(tokens * compress_out_dim * self.a_byte * 2),
                    store_kv_cache=int(compressed_tokens * compress_out_dim * self.kv_byte),
                ),
            ),
        ]
        return ops

    # ---- sparse attention (V4) ----

    def sparse_attention(self, layer_idx: int, step: StepShape,
                          compress_ratio: int) -> AttentionOp:
        """V4 fused_sparse_attention. attended = window + compressed + index_topk(if ratio==4) + 1 sink."""
        m = self.model
        tp = self.deploy.tp_size
        heads_per_tp = m.num_heads // tp
        head_dim = m.head_dim
        window = m.window_size
        index_topk = m.index_topk if compress_ratio == 4 else 0

        if step.phase == "decode":
            ctx_len = step.avg_decode_context_len
            bs = step.num_decode_requests
            all_compressed = ctx_len // compress_ratio if compress_ratio > 0 else 0
            if compress_ratio > 0 and index_topk > 0:
                compressed_attended = min(index_topk, all_compressed)
            else:
                compressed_attended = all_compressed
            attended = min(ctx_len, window) + compressed_attended + 1   # +1 sink

            qk_ops = attended * head_dim * heads_per_tp * bs * 2
            sv_ops = 1 * head_dim * attended * heads_per_tp * bs * 2
            softmax_ops = bs * heads_per_tp * attended * 1 * 5

            q_io = int(1 * head_dim * bs * heads_per_tp * self.a_byte)
            o_io = int(1 * head_dim * bs * heads_per_tp * self.a_byte)
            if self.hw.onchip_buffer > 0:
                block_size_r = max(1, math.floor(self.hw.onchip_buffer / (self.kv_byte * head_dim)))
                n_blocks_r = math.ceil(attended / block_size_r)
            else:
                n_blocks_r = 1
            kv_io = int(n_blocks_r * attended * head_dim * bs * self.kv_byte)

            subtype = "decode"
            num_tokens = bs
            num_seqs = bs
            q_len = 1
            kv_len = ctx_len
            flops = qk_ops + sv_ops + softmax_ops
            load_act = q_io
            store_act = int(o_io * 2)
        elif step.phase == "prefill":
            seqlen = step.max_prefill_seqlen
            bs = max(step.num_prefill_requests, 1)
            total_attended = 0
            total_kv_positions_loaded = 0
            if self.hw.onchip_buffer > 0:
                block_size_r = max(1, math.floor(self.hw.onchip_buffer / (self.kv_byte * head_dim)))
            else:
                block_size_r = None
            for pos in range(seqlen):
                local = min(pos + 1, window)
                all_compressed = (pos + 1) // compress_ratio if compress_ratio > 0 else 0
                if compress_ratio > 0 and index_topk > 0:
                    compressed_attended = min(index_topk, all_compressed)
                else:
                    compressed_attended = all_compressed
                attended_pos = local + compressed_attended + 1   # +1 sink
                total_attended += attended_pos
                if block_size_r is None:
                    total_kv_positions_loaded += attended_pos
                else:
                    n_blocks_r = math.ceil(attended_pos / block_size_r)
                    total_kv_positions_loaded += n_blocks_r * attended_pos

            qk_ops = total_attended * head_dim * heads_per_tp * bs * 2
            sv_ops = total_attended * head_dim * heads_per_tp * bs * 2
            softmax_ops = bs * heads_per_tp * total_attended * 5
            q_numel = seqlen * head_dim * bs * heads_per_tp * self.a_byte
            o_numel = seqlen * head_dim * bs * heads_per_tp * self.a_byte
            kv_io = int(total_kv_positions_loaded * head_dim * bs * self.kv_byte)

            subtype = "prefill"
            num_tokens = bs * seqlen
            num_seqs = bs
            q_len = seqlen
            kv_len = seqlen
            flops = qk_ops + sv_ops + softmax_ops
            load_act = int(q_numel)
            store_act = int(o_numel * 2)
        else:
            raise NotImplementedError(
                f"V4 sparse_attention 只支持 prefill/decode, got {step.phase!r}"
            )

        return AttentionOp(
            name=f"layer{layer_idx}_fused_sparse_attention",
            op_kind="attention", op_subtype=subtype,
            phase=step.phase, layer_idx=layer_idx, dtype="bf16",
            tags=("v4_sparse",),
            shape_fields={
                "num_tokens": num_tokens, "num_seqs": num_seqs,
                "q_len": q_len, "kv_len": kv_len,
                "num_q_heads": heads_per_tp, "num_kv_heads": 1,
                "head_dim": head_dim,
                "window_size": window,
                "compress_ratio": compress_ratio,
                "index_topk": index_topk,
            },
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields={
                **make_runtime(self.deploy, kernel_source="vllm_v4_sparse_attention"),
                "attention_backend": "v4_sparse",
                "kv_dtype": "bf16",
                "block_size": self.deploy.block_size,
            },
            formula_value=OperatorFormula(
                op_category="attention",
                flops=int(flops),
                load_act=load_act,
                store_act=store_act,
                load_kv_cache=kv_io,
            ),
        )

    # ---- wo_a / wo_b (low-rank O projection) ----

    def wo_a(self, layer_idx: int, tokens: int, phase: str) -> FormulaOp:
        m = self.model
        tp = self.deploy.tp_size
        heads_per_tp = m.num_heads // tp
        n_local_groups = max(m.o_groups // tp, 1)
        wo_a_ic = heads_per_tp * m.head_dim // n_local_groups
        wo_a_oc = n_local_groups * m.o_lora_rank
        op_prec = "fp8" if self.w_byte <= 1.0 else "fp16"
        return FormulaOp(
            name=f"layer{layer_idx}_wo_a",
            op_kind="gemm", op_subtype="wo_a",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"m": tokens, "n": wo_a_oc, "k": wo_a_ic},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_column_parallel_linear"),
            formula_value=OperatorFormula(
                op_category="matmul",
                flops=2 * tokens * wo_a_ic * wo_a_oc,
                load_weight=int(wo_a_ic * wo_a_oc * self.w_byte),
                load_act=int(tokens * wo_a_ic * self.a_byte),
                store_act=int(tokens * wo_a_oc * self.a_byte),
                op_precision=op_prec,
            ),
        )

    def wo_b(self, layer_idx: int, tokens: int, phase: str) -> FormulaOp:
        m = self.model
        tp = self.deploy.tp_size
        n_local_groups = max(m.o_groups // tp, 1)
        wo_b_ic = n_local_groups * m.o_lora_rank
        op_prec = "fp8" if self.w_byte <= 1.0 else "fp16"
        return FormulaOp(
            name=f"layer{layer_idx}_wo_b",
            op_kind="gemm", op_subtype="wo_b",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape_fields={"m": tokens, "n": m.hidden_dim, "k": wo_b_ic},
            parallel_fields=dense_parallel(self.deploy),
            runtime_fields=make_runtime(self.deploy, kernel_source="vllm_row_parallel_linear"),
            formula_value=OperatorFormula(
                op_category="matmul",
                flops=2 * tokens * wo_b_ic * m.hidden_dim,
                load_weight=int(wo_b_ic * m.hidden_dim * self.w_byte),
                load_act=int(tokens * wo_b_ic * self.a_byte),
                store_act=int(tokens * m.hidden_dim * self.a_byte),
                op_precision=op_prec,
            ),
        )
