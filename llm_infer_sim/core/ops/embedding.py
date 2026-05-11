"""Embedding and LM head operators."""

from llm_infer_sim.core.ops.base import OperatorProfile


def embedding(
    tokens: int,
    vocab_size: int,
    hidden_dim: int,
    w_byte: float,
    a_byte: float,
) -> OperatorProfile:
    """Token embedding lookup (pure memory access)."""
    return OperatorProfile(
        name="embedding",
        op_category="embedding",
        flops=0,
        load_weight=int(vocab_size * hidden_dim * w_byte),
        store_act=int(tokens * hidden_dim * a_byte),
    )


def lm_head(
    tokens: int,
    vocab_size: int,
    hidden_dim: int,
    tp_size: int,
    w_byte: float,
    a_byte: float,
) -> OperatorProfile:
    """LM head projection: [tokens, h] -> [tokens, vocab_size]."""
    oc = vocab_size // tp_size
    return OperatorProfile(
        name="lm_head",
        op_category="matmul",
        flops=2 * tokens * hidden_dim * oc,
        load_weight=int(hidden_dim * oc * w_byte),
        load_act=int(tokens * hidden_dim * a_byte),
        store_act=int(tokens * oc * a_byte),
    )
