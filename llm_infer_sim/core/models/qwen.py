"""QwenModelGraphTemplate — IMPL_PLAN §1.4 Step 1.9 + §4 (Stage 3a MoE).

阶段 1: Qwen3 dense, 每层:
    attn_norm → qkv_proj → rope → attention → o_proj → attn_add
        → mlp_norm → gate_up_proj → mlp_act → down_proj → mlp_add

阶段 3a: Qwen3 MoE (e.g. Qwen3-30B-A3B), MoE 层把 FFN 部分换成:
    mlp_norm → moe_gate
        → [ep>1: ep_alltoall_dispatch]
        → routed_experts
        → [ep=1 & tp>1: routed_expert_allreduce]
        → [ep>1:        ep_alltoall_combine]
        → [num_shared_experts>0: shared_expert_up_gate / _act / _down / _allreduce]
        → mlp_add

attention block 跟 dense 层相同. attn_allreduce (TP>1 后) 暂未注入, Step 3e 时统一处理.

全图: embedding → [layer ops × num_layers] → lm_head

暂不支持:
    HC (V4)                — 3c
    MLA / sparse           — 3b / 3c
    mixed step              — StepShape 不让进
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.operators.factories import FactoryBundle
from llm_infer_sim.core.operators.specs import Operator
from llm_infer_sim.core.profiles.model_config import ModelConfig


@dataclass(frozen=True)
class QwenModelGraphTemplate:
    model: ModelConfig

    def build_step(
        self,
        step: StepShape,
        factories: FactoryBundle,
    ) -> StepOpPlan:
        ops: list[Operator] = []
        tokens = step.total_tokens
        phase = step.phase

        ops.append(factories.embedding.embedding(tokens, phase))

        for layer_idx in range(self.model.num_layers):
            if self.model.is_moe_layer(layer_idx):
                ops.extend(self._build_moe_layer(layer_idx, step, factories))
            else:
                ops.extend(self._build_layer(layer_idx, step, factories))

        if phase == "prefill":
            head_tokens = max(step.num_prefill_requests, 1)
        else:
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

    # ---- dense layer ----

    def _build_layer(
        self,
        layer_idx: int,
        step: StepShape,
        factories: FactoryBundle,
    ) -> list[Operator]:
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

    # ---- MoE layer (3a) ----

    def _build_moe_layer(
        self,
        layer_idx: int,
        step: StepShape,
        factories: FactoryBundle,
    ) -> list[Operator]:
        if factories.moe is None or factories.collective is None:
            raise ValueError(
                "MoE layer requires FactoryBundle.moe + .collective; "
                "build engine via build_qwen_moe_roofline_engine() or pass them explicitly."
            )
        tokens = step.total_tokens
        phase = step.phase
        deploy = factories.moe.deploy
        tp = deploy.tp_size
        ep = deploy.ep_size
        h = self.model.hidden_dim
        a_byte = factories.moe.a_byte
        comm_bytes_h = int(tokens * h * a_byte)

        ops: list[Operator] = [
            # attention block (与 dense 层相同)
            factories.norm.attn_norm(layer_idx, tokens, phase),
            factories.dense.qkv_proj(layer_idx, tokens, phase),
            factories.attention.rope(layer_idx, tokens, phase),
            factories.attention.attention(layer_idx, step),
            factories.dense.o_proj(layer_idx, tokens, phase),
            factories.norm.attn_add(layer_idx, tokens, phase),
            # MoE FFN
            factories.norm.mlp_norm(layer_idx, tokens, phase),
            factories.moe.moe_gate(layer_idx, tokens, phase),
        ]

        if ep > 1:
            ops.append(factories.collective.alltoall(
                name="ep_alltoall_dispatch",
                message_bytes=comm_bytes_h,
                phase=phase, layer_idx=layer_idx, world_size=ep,
            ))

        ops.append(factories.moe.routed_experts(layer_idx, tokens, phase))

        if ep == 1 and tp > 1:
            ops.append(factories.collective.allreduce(
                name="routed_expert_allreduce",
                message_bytes=comm_bytes_h,
                phase=phase, layer_idx=layer_idx, world_size=tp,
            ))
        elif ep > 1:
            ops.append(factories.collective.alltoall(
                name="ep_alltoall_combine",
                message_bytes=comm_bytes_h,
                phase=phase, layer_idx=layer_idx, world_size=ep,
            ))

        if self.model.num_shared_experts > 0:
            ops.append(factories.moe.shared_expert_up_gate(layer_idx, tokens, phase))
            ops.append(factories.moe.shared_expert_act(layer_idx, tokens, phase))
            ops.append(factories.moe.shared_expert_down(layer_idx, tokens, phase))
            if tp > 1:
                ops.append(factories.collective.allreduce(
                    name="shared_expert_allreduce",
                    message_bytes=comm_bytes_h,
                    phase=phase, layer_idx=layer_idx, world_size=tp,
                ))

        ops.append(factories.norm.mlp_add(layer_idx, tokens, phase))
        return ops
