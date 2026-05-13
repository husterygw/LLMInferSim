"""Layer / model time builders (cost_model layer).

Source: 复制自 llm-viewer models/parallel.py (拆分: ModelConfig 留 profiles/model_config.py)
"""

from dataclasses import dataclass
from llm_infer_sim.core.ops.base import OperatorProfile
from llm_infer_sim.core.ops.linear import (
    linear_layer,
    fused_qkv_gemm,
    fused_gate_up_gemm,
)
from llm_infer_sim.core.ops.attention import (
    attention_decode_standard,
    attention_decode_flash,
    attention_prefill_standard,
    attention_prefill_flash,
    attention_prefill_sparse,
    attention_decode_sparse,
    rope_kernel,
)
from llm_infer_sim.core.ops.normalization import norm_layer, residual_add, mlp_activation, hc_pre, hc_post
from llm_infer_sim.core.ops.communication import allreduce_time, alltoall_time
from llm_infer_sim.core.ops.embedding import embedding, lm_head
from llm_infer_sim.core.cost_model.roofline import RooflineAnalyzer
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig




@dataclass
class LayerResult:
    """Timing result for one layer."""
    layer_idx: int
    layer_type: str  # "dense" | "moe"
    ops: list[OperatorProfile]
    t_compute: float   # sum of compute/memory time
    t_comm: float      # sum of communication time
    t_total: float     # t_compute + t_comm


def _comm_op(name: str, comm_bytes: float, comm_type: str) -> OperatorProfile:
    return OperatorProfile(
        name=name, op_category="communication",
        comm_bytes=comm_bytes, comm_type=comm_type,
    )


def _build_attention_block(
    stage: str,
    tokens: int,
    ctx_len: int,
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    layer_idx: int = 0,
) -> list[OperatorProfile]:
    """Build attention sub-block operators (shared by dense and MoE layers)."""
    tp = deploy.tp
    ops = []

    h = model.hidden_dim
    heads_per_tp = model.num_heads // tp
    kv_heads_per_tp = model.num_kv_heads // tp
    head_dim = model.head_dim

    # ── V4 path: uses sparse attention (window_size > 0) and grouped O projection ──
    is_v4 = model.window_size > 0 and model.o_groups > 0

    if is_v4:
        # HC pre merges HC copies before attention, but attention still applies attn_norm.
        if model.hc_mult > 0:
            ops.append(hc_pre("hc_attn_pre", tokens, h, model.hc_mult,
                              model.hc_sinkhorn_iters, deploy.a_byte, deploy.w_byte))
        ops.append(norm_layer("attn_norm", tokens, h, deploy.a_byte))

        # Determine per-op precision: FP8 for non-expert linear ops when global quant is FP8
        op_prec = "fp8" if deploy.w_byte <= 1.0 else "fp16"

        # 9-γ fusion 1: 真实 V4 用 fused_wqa_wkv (MergedColumnParallelLinear, disable_tp=True)
        # 一次性输出 [q_lora_rank, head_dim] concat; 旧拆分代码 x 读两次, fused 后只读一次.
        wqa_wkv_oc = model.q_lora_rank + head_dim
        ops.append(OperatorProfile(
            name="fused_wqa_wkv", op_category="matmul",
            flops=h * wqa_wkv_oc * tokens * 2,
            load_weight=int(h * wqa_wkv_oc * deploy.w_byte),
            load_act=int(h * tokens * deploy.a_byte),         # x 一次
            store_act=int(wqa_wkv_oc * tokens * deploy.a_byte),
            op_precision=op_prec,
        ))
        # q_norm + wq_b (norm 在 q_lora_rank 上, 然后 ColumnParallel q_b_proj)
        ops.append(norm_layer("q_norm", tokens, model.q_lora_rank, deploy.a_byte))
        ops.append(linear_layer("wq_b", model.q_lora_rank, heads_per_tp * head_dim, tokens,
                                deploy.w_byte, deploy.a_byte, deploy.kv_byte,
                                op_precision=op_prec))
        # kv_norm (在 head_dim 上, 不切 TP)
        # 阶段 9 fix 12: 真实 V4 用 fused_q_kv_rmsnorm 把 q_norm + kv_norm 融合到单 kernel.
        # 我们的 cost model 把它们当两个独立 norm op, 主要损失是一次 kernel launch overhead
        # (~µs/layer). hw.kernel_overhead 字典是 placeholder=0 时这个差异不可见,
        # 阶段 X calibration 填 kernel_overhead 后自动包含. 此处不单独融合.
        ops.append(norm_layer("kv_norm", tokens, head_dim, deploy.a_byte))

        # Compressor (layers with compress_ratio > 0)
        compress_ratio = model.get_compress_ratio(layer_idx)
        if compress_ratio > 0:
            # compress_wkv: Linear(dim → head_dim), compress_gate: Linear(dim → head_dim)
            # Runtime behavior in DeepSeek-V4:
            # - weights are stored as FP32 parameters
            # - input x is read in upstream activation dtype, then cast to FP32
            # - outputs kv/score are FP32 tensors

            coff = 2 if compress_ratio == 4 else 1
            compress_out_dim = coff * head_dim

            # 9-γ fusion 2: 真实 V4 用 fused_wkv_wgate (MergedColumnParallelLinear,
            # disable_tp=True, bf16 unquant), 一次输出 [coff*hd, coff*hd] concat.
            # 旧拆分代码 x 读两次, fused 后只读一次.
            fused_oc = 2 * compress_out_dim
            ops.append(OperatorProfile(
                name="fused_compress_wkv_wgate", op_category="matmul",
                flops=h * fused_oc * tokens * 2,
                load_weight=int(h * fused_oc * 2.0),    # bf16 weight
                load_act=int(h * tokens * deploy.a_byte),  # x 一次
                store_act=int(fused_oc * tokens * 2.0),     # bf16 output
                op_precision="fp16",
            ))
            # Softmax pooling: compress_ratio elements per group, ~5 ops/element.
            # 9-γ fusion 9 修: store 真实写 paged kv_cache(fp8/fp4), 不是 activation;
            # 用 deploy.kv_byte 替代 a_byte (compress 后的紧凑 KV).
            compressed_tokens = tokens // compress_ratio if compress_ratio > 0 else 0
            ops.append(OperatorProfile(
                name="compress_pool", op_category="activation",
                flops=tokens * compress_out_dim * 5,
                load_act=int(tokens * compress_out_dim * deploy.a_byte * 2),
                store_kv_cache=int(compressed_tokens * compress_out_dim * deploy.kv_byte),
            ))

            # Indexer (only for compress_ratio == 4 layers)
            if compress_ratio == 4 and model.index_topk > 0:
                # Inner compressor for indexer uses index_head_dim (128), NOT main head_dim (512).
                # Source: Indexer.__init__ → Compressor(args, ratio, self.head_dim=index_head_dim)
                # 9-γ fusion 3: indexer.compressor 同样 fused_wkv_wgate (bf16, disable_tp)
                idx_coff = 2  # always overlap (ratio==4)
                idx_compress_out_dim = idx_coff * model.index_head_dim
                idx_fused_oc = 2 * idx_compress_out_dim
                ops.append(OperatorProfile(
                    name="fused_index_compress_wkv_wgate", op_category="matmul",
                    flops=h * idx_fused_oc * tokens * 2,
                    load_weight=int(h * idx_fused_oc * 2.0),
                    load_act=int(h * tokens * deploy.a_byte),
                    store_act=int(idx_fused_oc * tokens * 2.0),
                    op_precision="fp16",
                ))
                # 9-β bug 2 修: index_wq_b 和 index_weights_proj 在真实 V4 里都是
                # ReplicatedLinear (disable_tp=True), 不切 TP. 每 rank 跑完整 index_n_heads.
                # 旧代码用 `index_n_heads // tp` 在 tp=8 时低估 8× weight read 和 compute.
                index_full_heads = model.index_n_heads  # NOT divided by tp
                ops.append(linear_layer("index_wq_b", model.q_lora_rank,
                                        index_full_heads * model.index_head_dim, tokens,
                                        deploy.w_byte, deploy.a_byte, deploy.kv_byte,
                                        op_precision=op_prec))
                # weights_proj (bf16, ReplicatedLinear): h → index_n_heads (no /tp).
                ops.append(linear_layer("index_weights_proj", h,
                                        index_full_heads, tokens,
                                        2.0, deploy.a_byte, deploy.kv_byte,
                                        op_precision="bf16"))
                # Scoring matmul: Q_index @ K_compressed → scores → topk.
                # Q size grows with `tokens` (queries this step); K size depends
                # on how many compressed entries already exist in the cache, i.e.
                # `cache_compressed_size = ctx_len // compress_ratio`. The previous
                # code used `tokens // ratio`, which collapses to 0 in decode
                # (tokens=batch_size=1) and silently drops the entire op.
                cache_compressed_size = ctx_len // compress_ratio if compress_ratio > 0 else 0
                # 9-β bug 4 修: indexer_kv_byte 从 deploy.indexer_kv_byte 读
                # (来自 vllm_config.attention_config.use_fp4_indexer_cache):
                #   fp8 默认 = 1.0 B/elem, fp4 = 0.5 B/elem. 旧代码写死 2.0 bf16 是错的.
                # 同时也修 bug 2 的 iH: load_act / store_act / flops 都用 index_full_heads
                # (不切 TP) 跟 ReplicatedLinear 一致.
                indexer_kv_byte = deploy.indexer_kv_byte
                ops.append(OperatorProfile(
                    name="index_score", op_category="attention",
                    flops=tokens * cache_compressed_size * model.index_head_dim * index_full_heads * 2,
                    load_act=int(tokens * index_full_heads * model.index_head_dim * deploy.a_byte),
                    load_kv_cache=int(cache_compressed_size * model.index_head_dim
                                      * tokens * indexer_kv_byte),
                    store_act=int(tokens * cache_compressed_size * index_full_heads * deploy.a_byte),
                ))

        # Sparse attention
        if stage == "decode":
            attn_ops = attention_decode_sparse(
                ctx_len, deploy.batch_size, heads_per_tp, head_dim,
                deploy.a_byte, deploy.kv_byte, model.window_size,
                compress_ratio, model.index_topk if compress_ratio == 4 else 0,
                onchip_buffer=hw.onchip_buffer,
            )
        else:
            bs = deploy.batch_size
            seq = tokens // bs if bs > 0 else tokens
            attn_ops = attention_prefill_sparse(
                seq, bs, heads_per_tp, head_dim,
                deploy.a_byte, deploy.kv_byte, model.window_size,
                compress_ratio, model.index_topk if compress_ratio == 4 else 0,
                onchip_buffer=hw.onchip_buffer,
            )
        ops.extend(attn_ops)

        # O path: wo_a (ColumnParallel) + wo_b (RowParallel)
        # wo_a: [n_local_heads * head_dim // n_local_groups, n_local_groups * o_lora_rank]
        n_local_groups = max(model.o_groups // tp, 1)
        wo_a_ic = heads_per_tp * head_dim // n_local_groups
        wo_a_oc = n_local_groups * model.o_lora_rank
        # 9-β bug 3 修: production vLLM 用 deepseek_v4_fp8_einsum("bhr,hdr->bhd")
        # 调 FP8 einsum (wo_a.weight + weight_scale_inv), 不是 BF16.
        # 旧注释说 "reference impl 用 BF16 for simplicity" 是 DeepSeek 参考实现的话,
        # 不是 vLLM production path. cost 模型应跟 production 一致 → 用 deploy.w_byte.
        ops.append(linear_layer("wo_a", wo_a_ic, wo_a_oc, tokens,
                                deploy.w_byte, deploy.a_byte, deploy.kv_byte,
                                op_precision=op_prec))
        # wo_b: RowParallel [n_local_groups * o_lora_rank, h]
        wo_b_ic = n_local_groups * model.o_lora_rank
        ops.append(linear_layer("wo_b", wo_b_ic, h, tokens,
                                deploy.w_byte, deploy.a_byte, deploy.kv_byte,
                                op_precision=op_prec))

        # TP AllReduce after wo_b (RowParallel)
        if tp > 1:
            ops.append(_comm_op("attn_allreduce",
                                tokens * h * deploy.a_byte, "allreduce"))

        # HC post replaces residual add
        if model.hc_mult > 0:
            ops.append(hc_post("hc_attn_post", tokens, h, model.hc_mult, deploy.a_byte))
        else:
            ops.append(residual_add("attn_add", tokens, h, deploy.a_byte))

        return ops

    # ── Standard / V3 MLA path ────────────────────────────────────────────
    # RMSNorm
    ops.append(norm_layer("attn_norm", tokens, h, deploy.a_byte))

    if model.kv_lora_rank > 0:
        # MLA path (DeepSeek-V3/V4, 详设 §4.1.4 + 阶段 8-β 修正)
        #
        # 真实 MLA 结构(对齐 vLLM `DeepseekV2Attention` impl):
        #   Q side (with optional LoRA decomposition):
        #     - q_lora_rank > 0: q_a_proj(h → q_lora_rank) + q_b_proj(q_lora_rank → heads×q_head_dim)
        #     - q_lora_rank == 0: q_proj(h → heads × q_head_dim)
        #   KV side(单个 fused down-projection):
        #     - kv_a_proj_with_mqa: h → kv_lora_rank + qk_rope_head_dim
        #     - kv_b_proj: kv_lora_rank → heads × (qk_nope_head_dim + v_head_dim)  (compute-time)
        #
        # 阶段 8-β 修正(从 8-α inspect 发现的 bug):
        #   旧代码用 q_proj/k_proj/v_proj 三个独立 Linear 加 dense head_dim(7168/128=56),
        #   完全没用 q_lora_rank 和真实 MLA 维度。V3 实际 q_head_dim=192/v_head_dim=128。
        qk_nope = model.qk_nope_head_dim if model.qk_nope_head_dim > 0 else head_dim
        qk_rope = model.rope_head_dim or 0  # = qk_rope_head_dim in V3 config
        v_dim = model.v_head_dim if model.v_head_dim > 0 else qk_nope  # MLA default = qk_nope
        q_head_dim = qk_nope + qk_rope  # 192 in V3 (128 nope + 64 rope)

        # Q projection: 可选 LoRA 分解
        if model.q_lora_rank > 0:
            ops.append(linear_layer("q_a_proj", h, model.q_lora_rank, tokens,
                                    deploy.w_byte, deploy.a_byte, deploy.kv_byte))
            ops.append(linear_layer("q_b_proj", model.q_lora_rank,
                                    heads_per_tp * q_head_dim, tokens,
                                    deploy.w_byte, deploy.a_byte, deploy.kv_byte))
        else:
            ops.append(linear_layer("q_proj", h, heads_per_tp * q_head_dim, tokens,
                                    deploy.w_byte, deploy.a_byte, deploy.kv_byte))

        # KV side fused down-projection (input proj that hits HBM only this much)
        # output: c_kv (size=kv_lora_rank) + k_rope (size=qk_rope_head_dim) per token
        ops.append(linear_layer("kv_a_proj_with_mqa", h,
                                model.kv_lora_rank + qk_rope, tokens,
                                deploy.w_byte, deploy.a_byte, deploy.kv_byte,
                                is_kv_proj=True))

        # kv_b_proj: c_kv → per-rank (qk_nope + v_head_dim) × heads, compute-time decompression
        kv_b_oc = heads_per_tp * (qk_nope + v_dim)
        ops.append(linear_layer("kv_b_proj", model.kv_lora_rank, kv_b_oc, tokens,
                                deploy.w_byte, deploy.a_byte, deploy.kv_byte))
    else:
        # 阶段 3: 标准 MHA/GQA 走 QKVParallelLinear fusion (详设 §4.7.1a (1))
        ops.append(fused_qkv_gemm(
            "qkv_proj",
            hidden=h,
            num_q_heads_per_tp=heads_per_tp,
            num_kv_heads_per_tp=kv_heads_per_tp,
            head_dim=head_dim,
            tokens=tokens,
            w_byte=deploy.w_byte,
            a_byte=deploy.a_byte,
            kv_byte=deploy.kv_byte,
        ))
        # RoPE: 独立 in-place kernel, 在 attention 之前 (详设 §4.7.1a (5))
        ops.append(rope_kernel(
            "rope",
            tokens=tokens,
            num_q_heads_per_tp=heads_per_tp,
            num_kv_heads_per_tp=kv_heads_per_tp,
            head_dim=head_dim,
            a_byte=deploy.a_byte,
        ))

    # MLA attention dimensions (阶段 8-β: v_head_dim 默认 qk_nope_head_dim, 不退回 head_dim)
    if model.kv_lora_rank > 0:
        _qk_nope = model.qk_nope_head_dim if model.qk_nope_head_dim > 0 else head_dim
        _qk_rope = model.rope_head_dim or (model.kv_latent_dim - model.kv_lora_rank)
        _v_dim = model.v_head_dim if model.v_head_dim > 0 else _qk_nope
        attn_qk_head_dim = _qk_nope + _qk_rope
        attn_v_head_dim = _v_dim
        o_proj_ic = heads_per_tp * _v_dim
    else:
        attn_qk_head_dim = None
        attn_v_head_dim = None
        o_proj_ic = heads_per_tp * head_dim

    # MLA decode parameters
    if model.kv_lora_rank > 0:
        decode_kv_latent = model.kv_latent_dim
        decode_kv_lora_rank = model.kv_lora_rank
    else:
        decode_kv_latent = None
        decode_kv_lora_rank = None

    # Attention
    if stage == "decode":
        if deploy.use_flash_attention:
            attn_ops = attention_decode_flash(
                ctx_len, deploy.batch_size, heads_per_tp, kv_heads_per_tp,
                head_dim, deploy.a_byte, deploy.kv_byte, hw.onchip_buffer,
                kv_latent_dim=decode_kv_latent, kv_lora_rank=decode_kv_lora_rank,
            )
        else:
            attn_ops = attention_decode_standard(
                ctx_len, deploy.batch_size, heads_per_tp, kv_heads_per_tp,
                head_dim, deploy.a_byte, deploy.kv_byte,
                kv_latent_dim=decode_kv_latent, kv_lora_rank=decode_kv_lora_rank,
            )
    else:
        bs = deploy.batch_size
        seq = tokens // bs if bs > 0 else tokens
        if deploy.use_flash_attention:
            attn_ops = attention_prefill_flash(
                seq, bs, heads_per_tp, kv_heads_per_tp,
                head_dim, deploy.a_byte, deploy.kv_byte, hw.onchip_buffer,
                qk_head_dim=attn_qk_head_dim, v_head_dim=attn_v_head_dim,
            )
        else:
            attn_ops = attention_prefill_standard(
                seq, bs, heads_per_tp, kv_heads_per_tp,
                head_dim, deploy.a_byte, deploy.kv_byte,
                qk_head_dim=attn_qk_head_dim, v_head_dim=attn_v_head_dim,
            )
    ops.extend(attn_ops)

    # O projection (Row Parallel: input dim / tp)
    ops.append(linear_layer("o_proj", o_proj_ic, h, tokens,
                            deploy.w_byte, deploy.a_byte, deploy.kv_byte))

    # TP AllReduce after attention
    if tp > 1:
        ops.append(_comm_op("attn_allreduce",
                            tokens * h * deploy.a_byte, "allreduce"))

    # Residual add
    ops.append(residual_add("attn_add", tokens, h, deploy.a_byte))

    return ops


def _build_dense_ffn_block(
    tokens: int,
    model: ModelConfig,
    deploy: DeployConfig,
) -> list[OperatorProfile]:
    """Build dense FFN sub-block operators."""
    tp = deploy.tp
    h = model.hidden_dim
    ffn_per_tp = model.ffn_dim // tp
    ops = []

    # V4 HC pre merges HC copies before FFN, but FFN still applies mlp_norm.
    if model.hc_mult > 0:
        ops.append(hc_pre("hc_ffn_pre", tokens, h, model.hc_mult,
                          model.hc_sinkhorn_iters, deploy.a_byte, deploy.w_byte))
    ops.append(norm_layer("mlp_norm", tokens, h, deploy.a_byte))

    # 阶段 3: Gate + Up 走 MergedColumnParallelLinear fusion (详设 §4.7.1a (2))
    ops.append(fused_gate_up_gemm(
        "gate_up_proj",
        hidden=h,
        intermediate_per_tp=ffn_per_tp,
        tokens=tokens,
        w_byte=deploy.w_byte,
        a_byte=deploy.a_byte,
    ))

    # Activation
    ops.append(mlp_activation("mlp_act", tokens, ffn_per_tp, deploy.a_byte))

    # Down (Row Parallel)
    ops.append(linear_layer("down_proj", ffn_per_tp, h, tokens,
                            deploy.w_byte, deploy.a_byte, deploy.kv_byte))

    # TP AllReduce after FFN
    if tp > 1:
        ops.append(_comm_op("mlp_allreduce",
                            tokens * h * deploy.a_byte, "allreduce"))

    # V4 HC post replaces residual add
    # NOTE: 详设 §4.7.1a (4) fused_add_rms_norm 算子已新增到 ops/normalization.py,
    # 但 layer_builder 沿用 (residual_add + 下层 attn_norm) 两 op 方案,
    # 因为 fusion 仅省 1 次 [tokens, hidden] 读取 (~µs/层),
    # 不值得引入跨层 ops 重排的复杂度。阶段 X calibration 触发时再评估。
    if model.hc_mult > 0:
        ops.append(hc_post("hc_ffn_post", tokens, h, model.hc_mult, deploy.a_byte))
    else:
        ops.append(residual_add("mlp_add", tokens, h, deploy.a_byte))

    return ops


def _build_moe_ffn_block(
    tokens: int,
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    moe_routing_skew: float = 0.0,
    layer_idx: int = -1,
) -> list[OperatorProfile]:
    """Build MoE FFN sub-block operators.

    Args:
        moe_routing_skew: 路由偏度 ∈ [0, 1] 用于 estimate_distinct_experts。
                          0=uniform (默认, 阶段 0-9 哲学), 1=极端 imbalance。
        layer_idx: 当前 layer 索引 (≥0 时生效), 用于 V4 hash MoE routing 判断:
                   layer_idx < model.num_hash_layers 的层走 tid2eid lookup (无 router GEMM),
                   FLOPs≈0. 默认 -1 表示不区分 (向后兼容).
    """
    tp = deploy.tp
    ep = deploy.ep
    h = model.hidden_dim
    ops = []

    # V4 HC pre merges HC copies before FFN, but FFN still applies mlp_norm.
    if model.hc_mult > 0:
        ops.append(hc_pre("hc_ffn_pre", tokens, h, model.hc_mult,
                          model.hc_sinkhorn_iters, deploy.a_byte, deploy.w_byte))
    ops.append(norm_layer("mlp_norm", tokens, h, deploy.a_byte))

    # Router gate (replicated, no comm).
    # 阶段 9 fix 13: V4 前 num_hash_layers 层用 tid2eid lookup (无 router GEMM),
    # FLOPs≈0; 跳过 moe_gate op 注入. 普通 MoE 模型 num_hash_layers=0 行为不变.
    is_hash_routed = (
        0 <= layer_idx < model.num_hash_layers if model.num_hash_layers > 0 else False
    )
    if not is_hash_routed:
        # 阶段 9 fix 14: V3 用 noaux_tc + softmax, V4 用 sqrtsoftplus(softplus(x)).
        # 两者 post-linear FLOPs 都 O(tokens × num_experts), 量级 << linear 部分
        # (linear = 2 × tokens × h × num_experts), 当前公式仅算 linear 部分 (主导项),
        # post-linear normalization 差异 < 0.5%, 不单独建模. 阶段 X 校准时如对不上数再细化.
        ops.append(OperatorProfile(
            name="moe_gate", op_category="matmul",
            flops=2 * tokens * h * model.num_experts,
            load_weight=int(h * model.num_experts * deploy.w_byte),
            load_act=int(tokens * h * deploy.a_byte),
            store_act=int(tokens * model.num_experts * deploy.a_byte),
            op_precision="fp32",
        ))
    else:
        # tid2eid lookup: 一次内存读 (token_id → expert_id 映射), FLOPs 几乎 0.
        # 14 sqrtsoftplus / softmax (非 hash 时) 比 softmax 多 ~5 ops/elem,
        # 量级 << linear, 不单独建模 (落 §10 阶段 9 显式不做).
        ops.append(OperatorProfile(
            name="moe_hash_lookup", op_category="activation",
            flops=0,
            load_act=int(tokens * 4),       # token_id index (int32 = 4 B)
            store_act=int(tokens * model.num_activated_experts * 4),  # expert_ids (int32)
        ))

    # AllToAll dispatch (EP group)
    if ep > 1:
        ops.append(_comm_op("ep_alltoall_dispatch",
                            tokens * h * deploy.a_byte, "alltoall"))

    # Routed expert FFN
    # Under pure TP (no EP): each expert is Column+Row Parallel sharded
    #   by tp, matching vLLM FusedMoE behavior (intermediate_size // tp).
    # Under EP: experts are distributed across ep devices, not TP-sharded.
    top_k = model.num_activated_experts
    expert_dim_per_device = model.expert_dim // tp if ep == 1 else model.expert_dim
    expert_flops = tokens * top_k * 3 * 2 * h * expert_dim_per_device // ep

    # Expert weight byte width: FP4 experts use 0.5 bytes/param
    expert_w_byte = 0.5 if model.expert_fp4 and hw.has_fp4_tc else deploy.w_byte

    # Expert compute precision: FP4 weights use FP4 TC on Blackwell, BF16 (dequant) on Hopper
    if model.expert_fp4:
        expert_precision = "fp4" if hw.has_fp4_tc else "fp8"
    else:
        expert_precision = ""  # follow global quantization settings

    # Roofline load_weight: per rank, only *distinct* activated experts are read.
    # 阶段 5-δ: 用 coupon collector (+ skew interp) 替换硬编码 top_k —— 因为单 step 内
    # tokens × top_k 路由总数往往覆盖比 top_k 多得多的 distinct experts。
    # tokens=1 + skew=0 时 distinct == top_k (退化到旧行为, decode 边界正确)。
    # 大 tokens + skew=0 时 distinct → num_experts (prefill 全 sweep, 真实贴近)。
    from llm_infer_sim.core.cost_model.moe_routing import estimate_distinct_experts
    distinct_experts = estimate_distinct_experts(
        tokens, top_k, model.num_experts, skew=moe_routing_skew
    )
    expert_weight_read = int(
        distinct_experts * 3 * h * expert_dim_per_device * expert_w_byte / ep
    )
    # Activation IO:
    # - TP only (ep=1): all tokens are replicated on every device; each token's
    #   activation is read once regardless of top_k (the dispatch is local).
    # - EP (ep>1): after AllToAll dispatch each device receives tokens*top_k/ep
    #   token copies, each of size h.
    tokens_per_device = tokens if ep == 1 else tokens * top_k // ep
    expert_act_in  = int(tokens_per_device * h * deploy.a_byte)
    expert_act_out = int(tokens_per_device * h * deploy.a_byte)

    # 阶段 9 fix 16: V4 MegaMoE 内部 FP4 quant + dispatch fusion 等 kernel-level 细节
    # 不改变 op-level 公式 (flops / weight read / act IO 跟拆开模型的 GEMM 等价).
    # 这里我们用单 routed_experts op 表达,精度差异在 op_precision (fp4/fp8) 体现.
    # MegaMoE specific 优化 (per-expert padding / sort-scatter overhead) 阶段 X 校准时
    # 通过 EfficiencyProfile (op_kind="moe_experts", shape_bucket=...) 反映.
    ops.append(OperatorProfile(
        name="routed_experts", op_category="matmul",
        flops=expert_flops,
        load_weight=expert_weight_read,
        load_act=expert_act_in,
        store_act=expert_act_out,
        op_precision=expert_precision,
    ))

    # TP AllReduce for routed experts (Row Parallel reduction, same as dense FFN)
    if ep == 1 and tp > 1:
        ops.append(_comm_op("routed_expert_allreduce",
                            tokens * h * deploy.a_byte, "allreduce"))

    # AllToAll combine (EP group)
    if ep > 1:
        ops.append(_comm_op("ep_alltoall_combine",
                            tokens * h * deploy.a_byte, "alltoall"))

    # Shared experts (if present) are replicated across EP ranks but still use
    # TP-sharded MLP layers in vLLM (MergedColumnParallelLinear + RowParallelLinear).
    # This means the shared FFN weights are partitioned by TP, and the final
    # shared expert output needs a TP all-reduce before it can be added to the
    # routed expert result.
    if model.num_shared_experts > 0:
        shared_dim = model.expert_dim * model.num_shared_experts
        shared_dim_per_device = shared_dim // tp

        # gate + up (TP-sharded column parallel)
        ops.append(OperatorProfile(
            name="shared_expert_up_gate", op_category="matmul",
            flops=2 * tokens * h * shared_dim_per_device * 2,
            load_weight=int(h * shared_dim_per_device * 2 * deploy.w_byte),
            load_act=int(tokens * h * deploy.a_byte),
            store_act=int(tokens * shared_dim_per_device * 2 * deploy.a_byte),
        ))

        # activation on local TP shard
        ops.append(OperatorProfile(
            name="shared_expert_act", op_category="activation",
            flops=5 * tokens * shared_dim_per_device,
            load_act=int(tokens * shared_dim_per_device * 2 * deploy.a_byte),
            store_act=int(tokens * shared_dim_per_device * deploy.a_byte),
        ))

        # down (TP-sharded row parallel)
        ops.append(OperatorProfile(
            name="shared_expert_down", op_category="matmul",
            flops=2 * tokens * shared_dim_per_device * h,
            load_weight=int(shared_dim_per_device * h * deploy.w_byte),
            load_act=int(tokens * shared_dim_per_device * deploy.a_byte),
            store_act=int(tokens * h * deploy.a_byte),
        ))

        if tp > 1:
            # 阶段 9 fix 15: 真实 V4 MegaMoE 模式下 reduce_results=True (RowParallel 内部
            # 触发 allreduce); 非 MegaMoE 模式下 reduce_results=False, allreduce 由 caller
            # 后做. 当前模型无 use_mega_moe knob, 默认按 MegaMoE 算 (即 reduce_results=True
            # 加 allreduce). 影响主要在非 MegaMoE V4 变体上, V4-Flash + EP 默认 MegaMoE 时
            # 当前行为正确. 阶段 X 校准时如果对不上数, 加 BackendExecutionProfile.use_mega_moe.
            ops.append(_comm_op("shared_expert_allreduce",
                                tokens * h * deploy.a_byte, "allreduce"))

    # V4 HC post replaces residual add
    if model.hc_mult > 0:
        ops.append(hc_post("hc_ffn_post", tokens, h, model.hc_mult, deploy.a_byte))
    else:
        ops.append(residual_add("mlp_add", tokens, h, deploy.a_byte))

    return ops


def _compute_layer_time(
    ops: list[OperatorProfile],
    hw: HardwareConfig,
    deploy: DeployConfig,
) -> tuple[float, float]:
    """Compute total time for a list of ops, returning (t_compute, t_comm)."""
    analyzer = RooflineAnalyzer(hw,
        w_bit=int(deploy.w_byte * 8), a_bit=int(deploy.a_byte * 8),
        kv_bit=int(deploy.kv_byte * 8))
    tp = deploy.tp
    ep = deploy.ep

    t_compute = 0.0
    t_comm = 0.0

    for op in ops:
        if op.op_category == "communication":
            if op.comm_type == "allreduce":
                t_comm += allreduce_time(op.comm_bytes, tp, hw)
            elif op.comm_type == "alltoall":
                t_comm += alltoall_time(op.comm_bytes, ep, hw)
        else:
            result = analyzer.analyze(op)
            t_compute += result.total_time

    return t_compute, t_comm


def dense_layer_time(
    layer_idx: int,
    stage: str,
    tokens: int,
    ctx_len: int,
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
) -> LayerResult:
    """Compute timing for a Dense transformer layer (§8.5.1)."""
    ops = []
    ops.extend(_build_attention_block(stage, tokens, ctx_len, model, deploy, hw, layer_idx))
    ops.extend(_build_dense_ffn_block(tokens, model, deploy))

    t_compute, t_comm = _compute_layer_time(ops, hw, deploy)

    return LayerResult(
        layer_idx=layer_idx,
        layer_type="dense",
        ops=ops,
        t_compute=t_compute,
        t_comm=t_comm,
        t_total=t_compute + t_comm,
    )


def moe_layer_time(
    layer_idx: int,
    stage: str,
    tokens: int,
    ctx_len: int,
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    moe_routing_skew: float = 0.0,
) -> LayerResult:
    """Compute timing for a MoE transformer layer (§8.5.2).

    moe_routing_skew: 路由偏度 ∈ [0, 1], 透传给 _build_moe_ffn_block。
    """
    ops = []
    ops.extend(_build_attention_block(stage, tokens, ctx_len, model, deploy, hw, layer_idx))
    ops.extend(_build_moe_ffn_block(tokens, model, deploy, hw, moe_routing_skew, layer_idx))

    t_compute, t_comm = _compute_layer_time(ops, hw, deploy)

    return LayerResult(
        layer_idx=layer_idx,
        layer_type="moe",
        ops=ops,
        t_compute=t_compute,
        t_comm=t_comm,
        t_total=t_compute + t_comm,
    )


def model_inference_time(
    stage: str,
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
    ctx_len: int = 0,
) -> tuple[float, list[LayerResult]]:
    """Full model inference time for one forward pass (§8.6.4).

    Args:
        stage: "prefill" or "decode"
        model: model architecture config
        deploy: deployment config (batch, quant, parallel)
        hw: hardware config
        ctx_len: context length (for decode, = input_len + decode_step)

    Returns:
        (total_time_seconds, list_of_layer_results)
    """
    if stage == "prefill":
        tokens = deploy.batch_size * deploy.input_len
        context = deploy.input_len
    else:
        tokens = deploy.batch_size
        context = ctx_len if ctx_len > 0 else deploy.input_len

    layer_results = []
    for i in range(model.num_layers):
        if model.is_moe_layer(i):
            lr = moe_layer_time(i, stage, tokens, context, model, deploy, hw)
        else:
            lr = dense_layer_time(i, stage, tokens, context, model, deploy, hw)
        layer_results.append(lr)

    total = sum(lr.t_total for lr in layer_results)

    # Embedding (prefill) + LM head
    tp = deploy.tp
    emb_op = embedding(tokens, model.vocab_size, model.hidden_dim,
                       deploy.w_byte, deploy.a_byte)
    head_op = lm_head(tokens, model.vocab_size, model.hidden_dim,
                      tp, deploy.w_byte, deploy.a_byte)

    analyzer = RooflineAnalyzer(hw,
        w_bit=int(deploy.w_byte * 8), a_bit=int(deploy.a_byte * 8),
        kv_bit=int(deploy.kv_byte * 8))
    total += analyzer.analyze(emb_op).total_time
    total += analyzer.analyze(head_op).total_time

    return total, layer_results


def compute_metrics(
    model: ModelConfig,
    deploy: DeployConfig,
    hw: HardwareConfig,
) -> dict:
    """Compute end-to-end inference metrics (TTFT, TPOT, throughput).

    Returns dict with keys: ttft_ms, tpot_ms, throughput_tps,
                            decode_step_times_ms
    """
    # Prefill → TTFT
    ttft, _ = model_inference_time("prefill", model, deploy, hw)

    # Decode → TPOT (average over output_len steps)
    decode_times = []
    for step in range(deploy.output_len):
        ctx = deploy.input_len + step
        t, _ = model_inference_time("decode", model, deploy, hw, ctx_len=ctx)
        decode_times.append(t)

    avg_tpot = sum(decode_times) / len(decode_times) if decode_times else 0.0

    # Throughput = dp_size × batch_size / TPOT
    dp = deploy.dp
    tps = dp * deploy.batch_size / avg_tpot if avg_tpot > 0 else float("inf")

    return {
        "ttft_ms": ttft * 1000,
        "tpot_ms": avg_tpot * 1000,
        "throughput_tps": tps,
        "decode_step_times_ms": [t * 1000 for t in decode_times],
    }
