"""MixedAttentionEstimator — 详设 §4.7.1b。

阶段 3: 仅实现 split_kernels 策略 (FlashAttention prefill + PagedAttention decode)。
其他 3 种策略 (unified_ragged / chunked_prefill_interleaved /
decode_priority_prefill_append) raise NotImplementedError, 推到 §10.5 子阶段。

设计原则:
  同一份 vLLM trace 在不同 attention backend 下成本不同, 必须由
  BackendExecutionProfile.mixed_attention.mode dispatch, 不在 cost model 写死。
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
        if mode in ("unified_ragged", "chunked_prefill_interleaved",
                    "decode_priority_prefill_append"):
            raise NotImplementedError(
                f"mixed_attention.mode={mode} 推到详设 §10.5 子阶段; "
                f"阶段 3 仅 split_kernels"
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
        """split_kernels: prefill / decode 走独立 kernel。"""
        tp = self.deploy.tp
        heads_per_tp = self.model.num_heads // tp
        kv_heads_per_tp = self.model.num_kv_heads // tp
        head_dim = self.model.head_dim

        # ---- prefill 段 ----
        t_prefill = 0.0
        if num_prefill_tokens > 0 and num_prefill_requests > 0:
            seq = max(max_prefill_seqlen, num_prefill_tokens // num_prefill_requests)
            if self.deploy.use_flash_attention:
                prefill_ops = attention_prefill_flash(
                    seqlen=seq,
                    batchsize=num_prefill_requests,
                    num_attention_heads=heads_per_tp,
                    num_key_value_heads=kv_heads_per_tp,
                    head_size=head_dim,
                    a_byte=self.deploy.a_byte,
                    kv_byte=self.deploy.kv_byte,
                    onchip_buffer=self.hw.onchip_buffer,
                )
            else:
                prefill_ops = attention_prefill_standard(
                    seqlen=seq,
                    batchsize=num_prefill_requests,
                    num_attention_heads=heads_per_tp,
                    num_key_value_heads=kv_heads_per_tp,
                    head_size=head_dim,
                    a_byte=self.deploy.a_byte,
                    kv_byte=self.deploy.kv_byte,
                )
            t_prefill = self._sum_op_times(prefill_ops)

        # ---- decode 段 ----
        t_decode = 0.0
        if num_decode_requests > 0:
            ctx = max(avg_decode_context_len, 1)
            if self.deploy.use_flash_attention:
                decode_ops = attention_decode_flash(
                    seqlen=ctx,
                    batchsize=num_decode_requests,
                    num_attention_heads=heads_per_tp,
                    num_key_value_heads=kv_heads_per_tp,
                    head_size=head_dim,
                    a_byte=self.deploy.a_byte,
                    kv_byte=self.deploy.kv_byte,
                    onchip_buffer=self.hw.onchip_buffer,
                )
            else:
                decode_ops = attention_decode_standard(
                    seqlen=ctx,
                    batchsize=num_decode_requests,
                    num_attention_heads=heads_per_tp,
                    num_key_value_heads=kv_heads_per_tp,
                    head_size=head_dim,
                    a_byte=self.deploy.a_byte,
                    kv_byte=self.deploy.kv_byte,
                )
            t_decode = self._sum_op_times(decode_ops)

        # ---- 合并 ----
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

    def _sum_op_times(self, ops: list[OperatorProfile]) -> float:
        total = 0.0
        for op in ops:
            res = self.analyzer.analyze(op)
            total += res.total_time
        return total
