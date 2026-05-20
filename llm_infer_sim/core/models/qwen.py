"""QwenModelGraphTemplate — IMPL_PLAN §1.4 Step 1.9.

阶段 1: Qwen3 dense, 每层 (按 §6.2 Qwen Dense Layer 顺序):
    attn_norm → qkv_proj → rope → attention → o_proj → attn_add
        → mlp_norm → gate_up_proj → mlp_act → down_proj → mlp_add
全图: embedding → [layer ops × num_layers] → lm_head

暂不支持:
    TP allreduce (collective)  - 阶段 5
    MoE                         - 阶段 4
    mixed step                  - StepShape 不让进
    MLA / sparse                - 后续阶段
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.graph.virtual_op import VirtualOp
from llm_infer_sim.core.ops.factories import FactoryBundle
from llm_infer_sim.core.profiles.model_config import ModelConfig


@dataclass(frozen=True)
class QwenModelGraphTemplate:
    model: ModelConfig

    def build_step(
        self,
        step: StepShape,
        factories: FactoryBundle,
    ) -> StepOpPlan:
        ops: list[VirtualOp] = []
        tokens = step.total_tokens
        phase = step.phase

        # ---- embedding (once per step) ----
        ops.append(factories.embedding.embedding(tokens, phase))

        # ---- per-layer ----
        for layer_idx in range(self.model.num_layers):
            ops.extend(self._build_layer(layer_idx, step, factories))

        # ---- lm_head: 每请求 1 个采样 token ----
        if phase == "prefill":
            head_tokens = max(step.num_prefill_requests, 1)
        else:  # decode
            head_tokens = step.num_decode_requests
        ops.append(factories.dense.lm_head(head_tokens, phase))

        return StepOpPlan(
            step_id=step.step_id,
            phase=phase,
            ops=tuple(ops),
            metadata={
                "model": self.model.name,
                "num_layers": self.model.num_layers,
                "execution_mode": step.execution_mode,
            },
        )

    def _build_layer(
        self,
        layer_idx: int,
        step: StepShape,
        factories: FactoryBundle,
    ) -> list[VirtualOp]:
        tokens = step.total_tokens
        phase = step.phase
        return [
            factories.norm.attn_norm(layer_idx, tokens, phase),
            factories.dense.qkv_proj(layer_idx, tokens, phase),
            factories.attention.rope(layer_idx, tokens, phase),
            factories.attention.attention(layer_idx, step),
            factories.dense.o_proj(layer_idx, tokens, phase),
            factories.norm.attn_add(layer_idx, tokens, phase),
            factories.norm.mlp_norm(layer_idx, tokens, phase),
            factories.dense.gate_up_proj(layer_idx, tokens, phase),
            factories.norm.mlp_act(layer_idx, tokens, phase),
            factories.dense.down_proj(layer_idx, tokens, phase),
            factories.norm.mlp_add(layer_idx, tokens, phase),
        ]
