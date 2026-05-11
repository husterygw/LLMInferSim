"""MoE FFN operators: router gate, routed experts, shared experts."""

from llm_infer_sim.core.ops.base import OperatorProfile


def moe_gate(
    tokens: int,
    hidden_dim: int,
    num_experts: int,
    w_byte: float,
    a_byte: float,
) -> OperatorProfile:
    """Router gate: [tokens, h] -> [tokens, num_experts] (replicated, no comm)."""
    return OperatorProfile(
        name="moe_gate",
        op_category="matmul",
        flops=2 * tokens * hidden_dim * num_experts,
        load_weight=int(hidden_dim * num_experts * w_byte),
        load_act=int(tokens * hidden_dim * a_byte),
        store_act=int(tokens * num_experts * a_byte),
    )


def routed_experts(
    tokens: int,
    hidden_dim: int,
    expert_dim: int,
    num_activated_experts: int,
    ep_size: int,
    w_byte: float,
    a_byte: float,
) -> OperatorProfile:
    """Routed expert FFN (SwiGLU: gate + up + down = 3 matmuls).

    FLOPs: tokens × top_k × 3 × 2 × h × expert_dim / ep
    load_weight (Roofline): top_k × 3 × h × expert_dim × w_byte
        (FusedMoE kernel reads only activated expert weights)
    Note: memory_per_device uses num_experts/ep for OOM check (see evaluation/parallel.py)
    """
    flops = tokens * num_activated_experts * 3 * 2 * hidden_dim * expert_dim // ep_size

    # Roofline load_weight = activated expert weights only
    weight_read = int(num_activated_experts * 3 * hidden_dim * expert_dim * w_byte)

    act_in = int(tokens * num_activated_experts * hidden_dim * a_byte // ep_size)
    act_out = int(tokens * num_activated_experts * hidden_dim * a_byte // ep_size)

    return OperatorProfile(
        name="routed_experts",
        op_category="matmul",
        flops=flops,
        load_weight=weight_read,
        load_act=act_in,
        store_act=act_out,
    )


def shared_experts_up_gate(
    tokens: int,
    hidden_dim: int,
    shared_dim: int,
    tp_size: int,
    w_byte: float,
    a_byte: float,
) -> OperatorProfile:
    """Shared expert gate + up projection (Column Parallel)."""
    dim_per_tp = shared_dim // tp_size
    return OperatorProfile(
        name="shared_expert_up_gate",
        op_category="matmul",
        flops=2 * tokens * hidden_dim * dim_per_tp * 2,  # gate + up
        load_weight=int(hidden_dim * dim_per_tp * 2 * w_byte),
        load_act=int(tokens * hidden_dim * a_byte),
        store_act=int(tokens * dim_per_tp * 2 * a_byte),
    )


def shared_experts_down(
    tokens: int,
    hidden_dim: int,
    shared_dim: int,
    tp_size: int,
    w_byte: float,
    a_byte: float,
) -> OperatorProfile:
    """Shared expert down projection (Row Parallel)."""
    dim_per_tp = shared_dim // tp_size
    return OperatorProfile(
        name="shared_expert_down",
        op_category="matmul",
        flops=2 * tokens * dim_per_tp * hidden_dim,
        load_weight=int(dim_per_tp * hidden_dim * w_byte),
        load_act=int(tokens * dim_per_tp * a_byte),
        store_act=int(tokens * hidden_dim * a_byte),
    )


def shared_experts_activation(
    tokens: int,
    shared_dim: int,
    tp_size: int,
    a_byte: float,
) -> OperatorProfile:
    """Shared expert SiLU + element_mul activation."""
    dim_per_tp = shared_dim // tp_size
    return OperatorProfile(
        name="shared_expert_act",
        op_category="activation",
        flops=5 * tokens * dim_per_tp,
        load_act=int(tokens * dim_per_tp * 2 * a_byte),
        store_act=int(tokens * dim_per_tp * a_byte),
    )
