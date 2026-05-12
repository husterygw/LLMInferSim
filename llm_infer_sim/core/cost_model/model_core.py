"""ModelCoreCostModel — 阶段 2: 切到 llm-viewer dense_layer_time / moe_layer_time。

阶段 2 重大改动 (vs 阶段 1):
  ─ 弃用阶段 1 手写的 _build_layer_ops (通用 dense decoder 简化模板, 漏 SwiGLU gate)
  ─ 改调 llm-viewer 的 dense_layer_time / moe_layer_time
  ─ ModelConfig 上的真实分支 (V3 MLA / V4 sparse / MoE) 都跟着免费来 ——
    阶段 5/8/9 大部分工作变成 "adapter profile_extractor 解析新字段", 而不是 "建 cost model"

能力清单:
  ✓ MHA / GQA dense decoder: pure prefill / pure decode
  ✓ SwiGLU FFN (gate + up + down 三个 GEMM, llm-viewer 的 _build_dense_ffn_block)
  ✓ MoE (dense_layer_time/moe_layer_time 自动按 ModelConfig.is_moe_layer 路由)
    —— 阶段 5 切 Qwen3-30B-A3B 时验
  ✓ MLA / V4 sparse —— ModelConfig 字段透传, 阶段 8/9 验

显式不做:
  - chunked prefill / mixed step (阶段 3)
  - TP/EP collective (cost model 已支持, 但阶段 4 才用真实 TP)
  - per-rank cost (symmetric ranks 假设, 阶段 6 才打破)
"""
from __future__ import annotations

from typing import Any

from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.ops.embedding import embedding, lm_head
from llm_infer_sim.core.cost_model.layer_builder import dense_layer_time, moe_layer_time
from llm_infer_sim.core.cost_model.roofline import RooflineAnalyzer
from llm_infer_sim.core.profiles.profile_manager import ProfileBundle
from llm_infer_sim.core.workload.workload import GlobalStepWorkload, StepPhase


class ModelCoreCostModel:
    """阶段 2: 调 llm-viewer builder 算 cost。"""

    def __init__(self, profile_bundle: ProfileBundle):
        self.bundle = profile_bundle
        self.hw = profile_bundle.hw
        self.eff = profile_bundle.efficiency
        self.model_cfg = profile_bundle.model
        self.analyzer = RooflineAnalyzer(
            self.hw,
            w_bit=self.eff.w_bit,
            a_bit=self.eff.a_bit,
            kv_bit=self.eff.kv_bit,
        )

    # ------- public API -------

    def estimate(self, workload: GlobalStepWorkload) -> dict[str, Any]:
        """按 workload 算 step cost (调 llm-viewer dense/moe_layer_time)。

        阶段 3:
          - PREFILL / DECODE: 沿用阶段 2 直接调 layer_builder 路径
          - MIXED / CHUNKED_PREFILL: 走 plan_builder + MixedAttentionEstimator
            (split_kernels 策略, 详设 §4.6.2 + §4.7.1b)
        """
        if workload.total_scheduled_tokens == 0:
            return self._empty()

        if workload.phase == StepPhase.PREFILL:
            return self._estimate_prefill(workload)
        if workload.phase == StepPhase.DECODE:
            return self._estimate_decode(workload)
        # MIXED / CHUNKED_PREFILL: 阶段 3 经 plan_builder 走 mixed cost path
        return self._estimate_mixed(workload)

    # ------- internals -------

    def _estimate_prefill(self, workload):
        num_tokens = workload.num_prefill_tokens
        batch = max(1, workload.num_prefill_requests)
        seqlen = max(1, num_tokens // batch)
        ctx_len = workload.max_context_len
        return self._per_step_cost("prefill", batch, seqlen, ctx_len)

    def _estimate_decode(self, workload):
        batch = max(1, workload.num_decode_requests)
        ctx_len = workload.max_context_len
        return self._per_step_cost("decode", batch, 1, ctx_len)

    def _estimate_mixed(self, workload):
        """阶段 3: mixed step 走 plan_builder + MixedAttentionEstimator。

        架构方案 α: model_core 内部调 plan_builder, estimate(workload) 接口不变。
        阶段 4 起接口升级到 estimate(workload, plan), plan_builder 调用前移到
        model_runner 层。
        """
        from llm_infer_sim.core.planning.plan_builder import build_mixed_plan

        plan = build_mixed_plan(workload, self.bundle)
        return self._cost_from_mixed_plan(workload, plan)

    def _cost_from_mixed_plan(self, workload, plan):
        """把 plan (dense layer_results + attention_override) 拍平成 cost dict。"""
        per_layer_breakdown: list[dict] = []
        per_op_breakdown: list[dict] = []
        total_compute = 0.0
        total_memory = 0.0
        total_comm = 0.0

        # ---- dense ops: 用 analyzer 拆 compute / memory 二栏, 跟 _per_step_cost 一致 ----
        for lr in plan.layer_results:
            t_layer_compute = 0.0
            t_layer_memory = 0.0
            for op in lr.ops:
                if op.op_category == "communication":
                    continue
                res = self.analyzer.analyze(op)
                t_layer_compute += res.t_compute
                t_layer_memory += res.t_memory
                op_time = max(res.t_compute, res.t_memory)
                per_op_breakdown.append({
                    "scope": f"layer{lr.layer_idx}",
                    "name": op.name,
                    "category": op.op_category,
                    "flops": op.flops,
                    "mem_bytes": op.mem_bytes,
                    "t_compute": res.t_compute,
                    "t_memory": res.t_memory,
                    "t_total": op_time,
                    "bound": res.bottleneck,
                })
            for op in lr.ops:
                if op.op_category == "communication":
                    per_op_breakdown.append({
                        "scope": f"layer{lr.layer_idx}",
                        "name": op.name,
                        "category": "communication",
                        "flops": 0,
                        "mem_bytes": int(op.comm_bytes),
                        "t_compute": 0.0,
                        "t_memory": 0.0,
                        "t_total": 0.0,
                        "bound": "communication",
                    })
            total_compute += t_layer_compute
            total_memory += t_layer_memory
            total_comm += lr.t_comm
            per_layer_breakdown.append({
                "layer_idx": lr.layer_idx,
                "layer_type": lr.layer_type,
                "t_compute": t_layer_compute,
                "t_memory": t_layer_memory,
                "t_comm": lr.t_comm,
                "t_total": lr.t_total,
            })

        # ---- attention 部分: 来自 MixedAttentionEstimator ----
        attn_total = 0.0
        attn_breakdown = {}
        if plan.attention_override is not None:
            attn_total = plan.attention_override["total_time"]
            attn_breakdown = plan.attention_override.get("breakdown", {})
            per_op_breakdown.append({
                "scope": "mixed_attention",
                "name": f"mixed_attn_{plan.attention_override['strategy']}",
                "category": "attention",
                "flops": 0,
                "mem_bytes": 0,
                "t_compute": attn_total,  # roofline 内已 max(compute, memory), 算 compute 列
                "t_memory": 0.0,
                "t_total": attn_total,
                "bound": "mixed",
            })

        # ---- Embedding + LM head ----
        tokens_for_stage = workload.total_scheduled_tokens
        batch_for_lm_head = max(1, workload.num_prefill_requests + workload.num_decode_requests)
        deploy_for_lm = self.bundle.deploy
        for scope, op in (
            ("embedding", embedding(
                tokens_for_stage,
                self.model_cfg.vocab_size,
                self.model_cfg.hidden_dim,
                self.eff.w_byte,
                self.eff.a_byte,
            )),
            ("lm_head", lm_head(
                batch_for_lm_head,
                self.model_cfg.vocab_size,
                self.model_cfg.hidden_dim,
                deploy_for_lm.tp,
                self.eff.w_byte,
                self.eff.a_byte,
            )),
        ):
            res = self.analyzer.analyze(op)
            total_compute += res.t_compute
            total_memory += res.t_memory
            op_time = max(res.t_compute, res.t_memory)
            per_op_breakdown.append({
                "scope": scope,
                "name": op.name,
                "category": op.op_category,
                "flops": op.flops,
                "mem_bytes": op.mem_bytes,
                "t_compute": res.t_compute,
                "t_memory": res.t_memory,
                "t_total": op_time,
                "bound": res.bottleneck,
            })

        # ---- 总时间 = roofline(dense) + attention(mixed) + comm + extra ----
        # dense 部分用 max(compute, memory), attention 已经是 roofline-合成的时间
        dense_time = max(total_compute, total_memory)
        total_time = dense_time + attn_total + total_comm + plan.extra_runtime_time

        attention_time_aggregated = attn_total + sum(
            r["t_total"] for r in per_op_breakdown if r["category"] == "attention"
            and r["scope"] != "mixed_attention"
        )
        linear_time = sum(
            r["t_total"] for r in per_op_breakdown if r["category"] == "matmul"
        )
        moe_time = sum(
            lr["t_compute"] for lr in per_layer_breakdown
            if lr.get("layer_type") == "moe"
        )

        if total_comm > max(dense_time, attn_total):
            bottleneck = "communication"
        elif attn_total > dense_time:
            bottleneck = "attention"
        elif total_compute > total_memory:
            bottleneck = "compute"
        elif total_memory > 0.0:
            bottleneck = "memory"
        else:
            bottleneck = "unknown"

        return {
            "total_time": total_time,
            "compute_time": total_compute,
            "memory_time": total_memory,
            "comm_time": total_comm,
            "attention_time": attention_time_aggregated,
            "linear_time": linear_time,
            "moe_time": moe_time,
            "bottleneck": bottleneck,
            "per_layer": per_layer_breakdown,
            "per_op": per_op_breakdown,
            "stage": "mixed",
            "tokens": tokens_for_stage,
            "batch": batch_for_lm_head,
            "ctx_len": workload.max_context_len,
            "mixed_breakdown": attn_breakdown,
            "mixed_strategy": plan.attention_override["strategy"] if plan.attention_override else "",
        }

    def _per_step_cost(self, stage: str, batch: int, seqlen: int, ctx_len: int):
        """走 llm-viewer dense_layer_time / moe_layer_time 链路。"""
        # 构造 step-specific DeployConfig (parallel + dtype 沿用 bundle, 覆盖 batch/seqlen)
        deploy = DeployConfig(
            batch_size=batch,
            input_len=seqlen,
            output_len=1,
            w_byte=self.eff.w_byte,
            a_byte=self.eff.a_byte,
            kv_byte=self.eff.kv_byte,
            parallel=self.bundle.deploy.parallel,
            use_flash_attention=self.bundle.deploy.use_flash_attention,
        )

        tokens_for_stage = batch * seqlen if stage == "prefill" else batch

        per_layer_breakdown: list[dict] = []
        per_op_breakdown: list[dict] = []
        total_compute = 0.0
        total_memory = 0.0
        total_comm = 0.0
        total_time = 0.0

        for layer_idx in range(self.model_cfg.num_layers):
            if self.model_cfg.is_moe_layer(layer_idx):
                lr = moe_layer_time(layer_idx, stage, tokens_for_stage,
                                    ctx_len, self.model_cfg, deploy, self.hw)
            else:
                lr = dense_layer_time(layer_idx, stage, tokens_for_stage,
                                      ctx_len, self.model_cfg, deploy, self.hw)

            # llm-viewer LayerResult: t_compute / t_comm / t_total / ops (op list)
            # llm-viewer 的 t_compute 实际是 sum(per-op result.total_time), 把 compute
            # 和 memory bound 的时间都算进去了 —— 我们重新拆细成 compute vs memory
            # 来填 breakdown 三栏。
            t_layer_compute = 0.0
            t_layer_memory = 0.0
            for op in lr.ops:
                if op.op_category == "communication":
                    continue  # comm 时间已在 lr.t_comm
                res = self.analyzer.analyze(op)
                t_layer_compute += res.t_compute
                t_layer_memory += res.t_memory
                op_time = max(res.t_compute, res.t_memory)
                per_op_breakdown.append(
                    {
                        "scope": f"layer{layer_idx}",
                        "name": op.name,
                        "category": op.op_category,
                        "flops": op.flops,
                        "mem_bytes": op.mem_bytes,
                        "t_compute": res.t_compute,
                        "t_memory": res.t_memory,
                        "t_total": op_time,
                        "bound": res.bottleneck,
                    }
                )

            # comm ops: 平摊到 per_op_breakdown (供 dump 时可见)
            for op in lr.ops:
                if op.op_category == "communication":
                    per_op_breakdown.append(
                        {
                            "scope": f"layer{layer_idx}",
                            "name": op.name,
                            "category": "communication",
                            "flops": 0,
                            "mem_bytes": int(op.comm_bytes),
                            "t_compute": 0.0,
                            "t_memory": 0.0,
                            # llm-viewer comm 单 op 时间分摊较复杂, 这里给 layer comm 的
                            # 平均值占位 (阶段 4 真 TP 时再细化)
                            "t_total": 0.0,
                            "bound": "communication",
                        }
                    )

            total_compute += t_layer_compute
            total_memory += t_layer_memory
            total_comm += lr.t_comm
            total_time += lr.t_total
            per_layer_breakdown.append(
                {
                    "layer_idx": lr.layer_idx,
                    "layer_type": lr.layer_type,
                    "t_compute": t_layer_compute,
                    "t_memory": t_layer_memory,
                    "t_comm": lr.t_comm,
                    "t_total": lr.t_total,
                }
            )

        # ----- Embedding + LM head (整模型一次) -----
        # 注: 用 max(t_compute, t_memory) 替代 llm-viewer result.total_time
        # 后者在 flops=0 (embedding lookup) 时退化为 0 (阶段 1 已记录此 bug)
        for scope, op in (
            ("embedding", embedding(
                tokens_for_stage,
                self.model_cfg.vocab_size,
                self.model_cfg.hidden_dim,
                self.eff.w_byte,
                self.eff.a_byte,
            )),
            ("lm_head", lm_head(
                batch,
                self.model_cfg.vocab_size,
                self.model_cfg.hidden_dim,
                deploy.tp,
                self.eff.w_byte,
                self.eff.a_byte,
            )),
        ):
            res = self.analyzer.analyze(op)
            total_compute += res.t_compute
            total_memory += res.t_memory
            op_time = max(res.t_compute, res.t_memory)
            total_time += op_time
            per_op_breakdown.append(
                {
                    "scope": scope,
                    "name": op.name,
                    "category": op.op_category,
                    "flops": op.flops,
                    "mem_bytes": op.mem_bytes,
                    "t_compute": res.t_compute,
                    "t_memory": res.t_memory,
                    "t_total": op_time,
                    "bound": res.bottleneck,
                }
            )

        # ----- 按 category 聚合 (详设 §4.7.1) -----
        attention_time = sum(
            r["t_total"] for r in per_op_breakdown if r["category"] == "attention"
        )
        linear_time = sum(
            r["t_total"] for r in per_op_breakdown if r["category"] == "matmul"
        )
        # MoE 层级时间: layer_type == "moe" 的 layer 走 moe_layer_time path
        # 阶段 2 dense 模型一直为 0; 阶段 5+ MoE 模型才非零
        moe_time = sum(
            lr["t_compute"] for lr in per_layer_breakdown
            if lr.get("layer_type") == "moe"
        )

        # bottleneck: roofline 下 latency = max(compute, memory) + comm,
        # 哪个组件占主导即 bottleneck
        if total_comm > max(total_compute, total_memory):
            bottleneck = "communication"
        elif total_compute > total_memory:
            bottleneck = "compute"
        elif total_memory > 0.0:
            bottleneck = "memory"
        else:
            bottleneck = "unknown"

        return {
            "total_time": total_time,
            "compute_time": total_compute,
            "memory_time": total_memory,
            "comm_time": total_comm,
            "attention_time": attention_time,
            "linear_time": linear_time,
            "moe_time": moe_time,
            "bottleneck": bottleneck,
            "per_layer": per_layer_breakdown,
            "per_op": per_op_breakdown,
            "stage": stage,
            "tokens": tokens_for_stage,
            "batch": batch,
            "ctx_len": ctx_len,
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "total_time": 0.0,
            "compute_time": 0.0,
            "memory_time": 0.0,
            "comm_time": 0.0,
            "attention_time": 0.0,
            "linear_time": 0.0,
            "moe_time": 0.0,
            "bottleneck": "unknown",
            "per_layer": [],
            "per_op": [],
            "stage": "empty",
            "tokens": 0,
            "batch": 0,
            "ctx_len": 0,
        }
