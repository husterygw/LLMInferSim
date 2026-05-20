"""AttentionOpFactory — V3 §6.5 + IMPL_PLAN §1.4 Step 1.8.

阶段 1 范围: Qwen GQA prefill / decode (FlashAttention-2). 公式从 core/ops/attention.py.
MLA / sparse / mixed 后续阶段补.
"""
from __future__ import annotations

from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.graph.virtual_op import VirtualOp
from llm_infer_sim.core.ops.attention import (
    attention_decode_flash,
    attention_prefill_flash,
    rope_kernel,
)
from llm_infer_sim.core.ops.factories._common import (
    dense_parallel,
    make_runtime,
    profile_to_formula,
)
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig


class AttentionOpFactory:
    def __init__(
        self,
        model: ModelConfig,
        deploy: DeployConfig,
        hw: HardwareConfig,
        *,
        a_byte: float = 2.0,
        kv_byte: float = 2.0,
    ):
        self.model = model
        self.deploy = deploy
        self.hw = hw
        self.a_byte = a_byte
        self.kv_byte = kv_byte

    def rope(self, layer_idx: int, tokens: int, phase: str) -> VirtualOp:
        tp = self.deploy.tp_size
        n_q = self.model.num_heads // tp
        n_kv = self.model.num_kv_heads // tp
        prof = rope_kernel(
            name="rope", tokens=tokens,
            num_q_heads_per_tp=n_q, num_kv_heads_per_tp=n_kv,
            head_dim=self.model.head_dim,
            a_byte=self.a_byte,
        )
        return VirtualOp(
            name=f"layer{layer_idx}_rope",
            op_kind="elementwise", op_subtype="rope",
            phase=phase, layer_idx=layer_idx, dtype="bf16",
            shape={
                "tokens": tokens,
                "num_q_heads": n_q,
                "num_kv_heads": n_kv,
                "head_dim": self.model.head_dim,
            },
            parallel=dense_parallel(self.deploy),
            runtime=make_runtime(self.deploy),
            formula=profile_to_formula(prof),
        )

    def attention(self, layer_idx: int, step: StepShape) -> VirtualOp:
        """GQA flash attention. prefill 假设 1 个请求 ISL tokens, decode 每请求 1 token + ctx_len."""
        tp = self.deploy.tp_size
        n_q = self.model.num_heads // tp
        n_kv = self.model.num_kv_heads // tp
        head_dim = self.model.head_dim

        if step.phase == "prefill":
            seqlen = step.max_prefill_seqlen
            bs = max(step.num_prefill_requests, 1)
            profs = attention_prefill_flash(
                seqlen=seqlen, batchsize=bs,
                num_attention_heads=n_q,
                num_key_value_heads=n_kv,
                head_size=head_dim,
                a_byte=self.a_byte, kv_byte=self.kv_byte,
                onchip_buffer=self.hw.onchip_buffer,
            )
            subtype = "prefill"
            q_len = seqlen
            kv_len = seqlen
            num_tokens = bs * seqlen
            num_seqs = bs
        elif step.phase == "decode":
            ctx_len = step.avg_decode_context_len
            bs = step.num_decode_requests
            profs = attention_decode_flash(
                seqlen=ctx_len, batchsize=bs,
                num_attention_heads=n_q,
                num_key_value_heads=n_kv,
                head_size=head_dim,
                a_byte=self.a_byte, kv_byte=self.kv_byte,
                onchip_buffer=self.hw.onchip_buffer,
            )
            subtype = "decode"
            q_len = 1
            kv_len = ctx_len
            num_tokens = bs
            num_seqs = bs
        else:
            raise NotImplementedError(
                f"AttentionOpFactory.attention 阶段 1 只支持 prefill / decode, got {step.phase!r}"
            )

        prof = profs[0]  # flash 返单 fused op

        return VirtualOp(
            name=f"layer{layer_idx}_attention",
            op_kind="attention", op_subtype=subtype,
            phase=step.phase, layer_idx=layer_idx, dtype="bf16",
            shape={
                "num_tokens": num_tokens, "num_seqs": num_seqs,
                "q_len": q_len, "kv_len": kv_len,
                "num_q_heads": n_q, "num_kv_heads": n_kv,
                "head_dim": head_dim,
            },
            parallel=dense_parallel(self.deploy),
            runtime={
                **make_runtime(self.deploy, kernel_source="vllm_flash_attn"),
                "attention_backend": "flash_attn",
                "kv_dtype": "bf16",
                "block_size": self.deploy.block_size,
            },
            formula=profile_to_formula(prof),
        )
