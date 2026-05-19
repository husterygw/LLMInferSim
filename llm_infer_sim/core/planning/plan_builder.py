"""DistributedPlanBuilder — 详设 §4.6.2。

阶段 3 范围:
  - 仅落 mixed step path: dense GEMM 走 merged (M=total_tokens) +
    attention 由 MixedAttentionEstimator 覆盖。
  - PREFILL/DECODE phase 不走本 builder, model_core 直接调 dense_layer_time。
  - mixed_mode 仅 split_kernels (经 backend.mixed_attention.mode dispatch)。
  - dense_gemm.mode=split 推到 §10.5。

阶段 4 起:
  - 全 phase 走 plan_builder, model_core.estimate(workload, plan) 接口升级。
  - plan_builder 落 per-rank op 列表 (含 TP/EP collective 真实分配)。
"""
from __future__ import annotations

from llm_infer_sim.core.cost_model.layer_builder import (
    LayerResult,
    dense_layer_time,
    moe_layer_time,
)
from llm_infer_sim.core.cost_model.mixed_attention import MixedAttentionEstimator
from llm_infer_sim.core.ops.base import OperatorProfile
from llm_infer_sim.core.cost_model.roofline import RooflineAnalyzer
from llm_infer_sim.core.planning.execution_plan import DistributedExecutionPlan
from llm_infer_sim.core.profiles.profile_manager import ProfileBundle
from llm_infer_sim.core.workload.workload import GlobalStepWorkload, StepPhase


def build_mixed_plan(
    workload: GlobalStepWorkload,
    bundle: ProfileBundle,
) -> DistributedExecutionPlan:
    """构造 mixed step 的执行计划 (详设 §4.6.2)。

    步骤:
      1. dense GEMM/FFN/comm: 用 merged-prefill 公式 (M = total_tokens, batch=1)
      2. 从每层 LayerResult 剥离 attention ops, 得到 dense-only LayerResult
      3. attention 由 MixedAttentionEstimator 单独算, 写入 attention_override
      4. extra_runtime_time: 阶段 3 暂为 0 (kernel launch overhead 推到 D 块校准)
    """
    if bundle.backend.dense_gemm.mode == "split":
        raise NotImplementedError(
            "dense_gemm.mode=split 推到详设 §10.5; 阶段 3 仅 merged"
        )

    model = bundle.model
    deploy_template = bundle.deploy
    hw = bundle.hw

    # 阶段 3 dense GEMM 用 merged 模式: 所有 token 合并为单 batch 的 prefill
    total_tokens = workload.total_scheduled_tokens
    if total_tokens == 0:
        return DistributedExecutionPlan(
            step_id=workload.step_id,
            mixed_mode=bundle.backend.mixed_attention.mode,
        )

    # 临时 DeployConfig: batch_size=1, input_len=total_tokens (merged GEMM 形状)
    deploy_merged = _override_deploy(deploy_template, batch_size=1, input_len=total_tokens)

    layer_results_dense: list[LayerResult] = []
    analyzer = RooflineAnalyzer(
        hw,
        w_bit=int(deploy_merged.w_byte * 8),
        a_bit=int(deploy_merged.a_byte * 8),
        kv_bit=int(deploy_merged.kv_byte * 8),
    )
    ctx_len = workload.max_context_len

    moe_routing = bundle.backend.moe_routing
    for layer_idx in range(model.num_layers):
        if model.is_moe_layer(layer_idx):
            lr = moe_layer_time(layer_idx, "prefill", total_tokens,
                                ctx_len, model, deploy_merged, hw,
                                moe_routing_skew=moe_routing.get_skew_for_layer(layer_idx))
        else:
            lr = dense_layer_time(layer_idx, "prefill", total_tokens,
                                  ctx_len, model, deploy_merged, hw)
        layer_results_dense.append(_strip_attention(lr, analyzer))

    # MixedAttentionEstimator 算 attention 部分
    estimator = MixedAttentionEstimator(
        model=model,
        hw=hw,
        deploy=deploy_merged,
        backend=bundle.backend,
        efficiency_profile=bundle.efficiency,    # B.6: per-op lookup 精化
    )
    attn_cost = estimator.estimate(
        num_prefill_tokens=workload.num_prefill_tokens,
        num_prefill_requests=workload.num_prefill_requests,
        num_decode_requests=workload.num_decode_requests,
        max_prefill_seqlen=workload.max_prefill_seqlen,
        avg_decode_context_len=workload.avg_decode_context_len,
    )

    return DistributedExecutionPlan(
        step_id=workload.step_id,
        mixed_mode=bundle.backend.mixed_attention.mode,
        layer_results=layer_results_dense,
        attention_override=attn_cost,
        extra_runtime_time=0.0,
    )


def _strip_attention(lr: LayerResult, analyzer: RooflineAnalyzer) -> LayerResult:
    """从 LayerResult 中剥离 attention kernel ops, 重算 t_compute。

    保留:
      - matmul (含 q/k/v_proj, o_proj, gate/up/down_proj, kv_b_proj)
      - norm / activation
      - communication (这里只是统计, t_comm 已算好)
    剥离:
      - op_category == "attention"  (qk_matmul / sv_matmul / softmax / fused_attention 等)
    """
    new_ops: list[OperatorProfile] = []
    new_t_compute = 0.0
    for op in lr.ops:
        if op.op_category == "attention":
            continue
        new_ops.append(op)
        if op.op_category != "communication":
            res = analyzer.analyze(op)
            new_t_compute += res.total_time
    return LayerResult(
        layer_idx=lr.layer_idx,
        layer_type=lr.layer_type,
        ops=new_ops,
        t_compute=new_t_compute,
        t_comm=lr.t_comm,
        t_total=new_t_compute + lr.t_comm,
    )


def _override_deploy(template, batch_size: int, input_len: int):
    """复制一份 DeployConfig, 覆盖 batch_size / input_len。

    output_len 不动, 因为 cost path 不依赖。parallel/dtype 沿用。
    """
    from llm_infer_sim.core.profiles.deploy import DeployConfig
    return DeployConfig(
        batch_size=batch_size,
        input_len=input_len,
        output_len=template.output_len,
        w_byte=template.w_byte,
        a_byte=template.a_byte,
        kv_byte=template.kv_byte,
        parallel=template.parallel,
        use_flash_attention=template.use_flash_attention,
        overlap_comm=template.overlap_comm,
    )
