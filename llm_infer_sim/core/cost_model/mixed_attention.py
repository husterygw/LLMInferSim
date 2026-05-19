"""MixedAttentionEstimator — 详设 §4.7.1b。

阶段 3:    实现 split_kernels (FlashAttention prefill + PagedAttention decode)。
阶段 3.5: 实现 unified_ragged (FA varlen / FlashInfer 单 kernel ragged batch)。

其他 2 种策略 (chunked_prefill_interleaved / decode_priority_prefill_append)
仍 raise NotImplementedError, 推到 §10.5 子阶段。

设计原则:
  同一份 vLLM trace 在不同 attention backend 下成本不同, 必须由
  BackendExecutionProfile.mixed_attention.mode dispatch, 不在 cost model 写死。
  mode 由 §4.8.1.1 adapters/vllm/profile_extractor._extract_backend_profile()
  从 vllm_config 推导 (core 完全框架无关, 详设 §1.1)。
"""
from __future__ import annotations

from llm_infer_sim.core.cost_model.roofline import RooflineAnalyzer
from llm_infer_sim.core.ops.attention import (
    attention_decode_flash,
    attention_decode_standard,
    attention_prefill_flash,
    attention_prefill_standard,
)
from llm_infer_sim.core.ops.base import OperatorProfile
from llm_infer_sim.core.profiles.backend_profile import BackendExecutionProfile
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig


class MixedAttentionEstimator:
    """根据 backend profile 估算 mixed batch 中 attention 部分的成本。

    入参以 GlobalStepWorkload 的聚合字段为准 (per-request 拆分推到 §10.5)。
    """

    def __init__(
        self,
        model: ModelConfig,
        hw: HardwareConfig,
        deploy: DeployConfig,
        backend: BackendExecutionProfile,
        efficiency_profile=None,   # EfficiencyProfile | None, B.6
    ) -> None:
        self.model = model
        self.hw = hw
        self.deploy = deploy
        self.backend = backend
        self.analyzer = RooflineAnalyzer(
            hw,
            w_bit=int(deploy.w_byte * 8),
            a_bit=int(deploy.a_byte * 8),
            kv_bit=int(deploy.kv_byte * 8),
            efficiency_profile=efficiency_profile,
        )

    def estimate(
        self,
        num_prefill_tokens: int,
        num_prefill_requests: int,
        num_decode_requests: int,
        max_prefill_seqlen: int,
        avg_decode_context_len: int,
    ) -> dict:
        """返回 mixed step 的 attention 时间估算。

        Returns:
            {
                "per_layer_time": float,    # 每层 attention 时间 (秒)
                "total_time": float,        # × num_layers
                "strategy": str,            # 策略名
                "breakdown": {...},         # 策略相关明细
            }
        """
        mode = self.backend.mixed_attention.mode
        if mode == "split_kernels":
            return self._split_kernels(
                num_prefill_tokens,
                num_prefill_requests,
                num_decode_requests,
                max_prefill_seqlen,
                avg_decode_context_len,
            )
        if mode == "unified_ragged":
            return self._unified_ragged(
                num_prefill_tokens,
                num_prefill_requests,
                num_decode_requests,
                max_prefill_seqlen,
                avg_decode_context_len,
            )
        if mode in ("chunked_prefill_interleaved",
                    "decode_priority_prefill_append"):
            raise NotImplementedError(
                f"mixed_attention.mode={mode} 推到详设 §10.5 子阶段; "
                f"阶段 3.5 仅 split_kernels / unified_ragged"
            )
        raise ValueError(f"Unknown mixed_attention.mode: {mode}")

    # ------- 策略实现 -------

    def _split_kernels(
        self,
        num_prefill_tokens: int,
        num_prefill_requests: int,
        num_decode_requests: int,
        max_prefill_seqlen: int,
        avg_decode_context_len: int,
    ) -> dict:
        """split_kernels: prefill / decode 走独立 kernel (PagedAttention v0 风格)。"""
        prefill_ops = self._build_prefill_ops(
            num_prefill_tokens, num_prefill_requests, max_prefill_seqlen
        )
        decode_ops = self._build_decode_ops(
            num_decode_requests, avg_decode_context_len
        )
        t_prefill = self._sum_op_times(prefill_ops)
        t_decode = self._sum_op_times(decode_ops)

        policy = self.backend.mixed_attention
        sync_time = policy.inter_kernel_sync_us * 1e-6
        if policy.prefill_decode_overlap:
            t_per_layer = max(t_prefill, t_decode) + sync_time
        else:
            t_per_layer = t_prefill + t_decode + sync_time

        return {
            "per_layer_time": t_per_layer,
            "total_time": t_per_layer * self.model.num_layers,
            "strategy": "split_kernels",
            "breakdown": {
                "t_prefill": t_prefill,
                "t_decode": t_decode,
                "sync_overhead": sync_time,
                "overlap": policy.prefill_decode_overlap,
            },
        }

    def _unified_ragged(
        self,
        num_prefill_tokens: int,
        num_prefill_requests: int,
        num_decode_requests: int,
        max_prefill_seqlen: int,
        avg_decode_context_len: int,
    ) -> dict:
        """unified_ragged: 单 kernel 处理 mixed varlen batch (FA varlen / FlashInfer)。

        阶段 3.5 设计要点 (详设 §4.7.1b):
          1. 用现有 attention_prefill_flash / attention_decode_flash 分别构造
             OperatorProfile, 保留 per-segment 真实 (q_len, kv_len, h_q, h_kv) 形状。
          2. 在 OperatorProfile 层聚合 flops + 5-way mem decomposition
             (而非时间层相加), 构造 1 个 merged op 跑 1 次 RooflineAnalyzer.analyze()
             —— 单 kernel = 单次 max(t_compute, t_memory) 的物理语义。
          3. 除以 ragged_efficiency (阶段 0-9 = 1.0, 与 EfficiencyProfile.placeholder
             同一哲学; SM 利用率函数推到阶段 X §9.4.2 microbench 拟合)。
          4. 不加 inter_kernel_sync_us (单 kernel 无 launch 间隔; 同样阶段 0-9 = 0,
             校准时跟 hw.kernel_overhead 字典统一进)。

        与 split_kernels 的差异: roofline 取 max 的次序 (纯结构性, 无未校准折扣)
          split    = max(t_c_pf, t_m_pf) + max(t_c_dc, t_m_dc) + sync   # sync=0
          unified  = max(t_c_pf + t_c_dc, t_m_pf + t_m_dc) / efficiency  # efficiency=1
        在 prefill compute-bound + decode memory-bound 不平衡时差异显著。
        """
        prefill_ops = self._build_prefill_ops(
            num_prefill_tokens, num_prefill_requests, max_prefill_seqlen
        )
        decode_ops = self._build_decode_ops(
            num_decode_requests, avg_decode_context_len
        )
        all_ops = list(prefill_ops) + list(decode_ops)

        if not all_ops:
            return {
                "per_layer_time": 0.0,
                "total_time": 0.0,
                "strategy": "unified_ragged",
                "breakdown": {"empty": True},
            }

        merged = _merge_ops(all_ops, name="unified_ragged_attn")
        res = self.analyzer.analyze(merged)

        eff = max(self.backend.mixed_attention.ragged_efficiency, 1e-6)
        t_per_layer = res.total_time / eff

        return {
            "per_layer_time": t_per_layer,
            "total_time": t_per_layer * self.model.num_layers,
            "strategy": "unified_ragged",
            "breakdown": {
                "merged_flops": merged.flops,
                "merged_mem_bytes": merged.mem_bytes,
                "t_compute": res.t_compute,
                "t_memory": res.t_memory,
                "ragged_efficiency": eff,
                "bottleneck": res.bottleneck,
            },
        }

    # ------- 共用工具 -------

    def _build_prefill_ops(
        self,
        num_prefill_tokens: int,
        num_prefill_requests: int,
        max_prefill_seqlen: int,
    ) -> list[OperatorProfile]:
        if num_prefill_tokens <= 0 or num_prefill_requests <= 0:
            return []
        seq = max(max_prefill_seqlen, num_prefill_tokens // num_prefill_requests)
        heads_per_tp = self.model.num_heads // self.deploy.tp
        kv_heads_per_tp = self.model.num_kv_heads // self.deploy.tp
        if self.deploy.use_flash_attention:
            return attention_prefill_flash(
                seqlen=seq,
                batchsize=num_prefill_requests,
                num_attention_heads=heads_per_tp,
                num_key_value_heads=kv_heads_per_tp,
                head_size=self.model.head_dim,
                a_byte=self.deploy.a_byte,
                kv_byte=self.deploy.kv_byte,
                onchip_buffer=self.hw.onchip_buffer,
            )
        return attention_prefill_standard(
            seqlen=seq,
            batchsize=num_prefill_requests,
            num_attention_heads=heads_per_tp,
            num_key_value_heads=kv_heads_per_tp,
            head_size=self.model.head_dim,
            a_byte=self.deploy.a_byte,
            kv_byte=self.deploy.kv_byte,
        )

    def _build_decode_ops(
        self,
        num_decode_requests: int,
        avg_decode_context_len: int,
    ) -> list[OperatorProfile]:
        if num_decode_requests <= 0:
            return []
        ctx = max(avg_decode_context_len, 1)
        heads_per_tp = self.model.num_heads // self.deploy.tp
        kv_heads_per_tp = self.model.num_kv_heads // self.deploy.tp
        if self.deploy.use_flash_attention:
            return attention_decode_flash(
                seqlen=ctx,
                batchsize=num_decode_requests,
                num_attention_heads=heads_per_tp,
                num_key_value_heads=kv_heads_per_tp,
                head_size=self.model.head_dim,
                a_byte=self.deploy.a_byte,
                kv_byte=self.deploy.kv_byte,
                onchip_buffer=self.hw.onchip_buffer,
            )
        return attention_decode_standard(
            seqlen=ctx,
            batchsize=num_decode_requests,
            num_attention_heads=heads_per_tp,
            num_key_value_heads=kv_heads_per_tp,
            head_size=self.model.head_dim,
            a_byte=self.deploy.a_byte,
            kv_byte=self.deploy.kv_byte,
        )

    def _sum_op_times(self, ops: list[OperatorProfile]) -> float:
        total = 0.0
        for op in ops:
            res = self.analyzer.analyze(op)
            total += res.total_time
        return total


def _merge_ops(ops: list[OperatorProfile], name: str) -> OperatorProfile:
    """把多个 attention OperatorProfile 合并成 1 个 (聚合 flops + 5-way mem)。

    用于 unified_ragged: 表达 "单 kernel 一次性处理所有 token" 的物理语义,
    分析时只做一次 max(t_compute, t_memory)。
    """
    return OperatorProfile(
        name=name,
        op_category="attention",
        flops=sum(op.flops for op in ops),
        load_weight=sum(op.load_weight for op in ops),
        load_act=sum(op.load_act for op in ops),
        store_act=sum(op.store_act for op in ops),
        load_kv_cache=sum(op.load_kv_cache for op in ops),
        store_kv_cache=sum(op.store_kv_cache for op in ops),
    )
