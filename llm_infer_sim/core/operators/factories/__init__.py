"""Operator factories."""
from dataclasses import dataclass

from llm_infer_sim.core.operators.factories.attention import AttentionOpFactory
from llm_infer_sim.core.operators.factories.communication import CollectiveOpFactory
from llm_infer_sim.core.operators.factories.dense import DenseOpFactory
from llm_infer_sim.core.operators.factories.embedding import EmbeddingOpFactory
from llm_infer_sim.core.operators.factories.indexer import IndexerOpFactory
from llm_infer_sim.core.operators.factories.moe import MoEOpFactory
from llm_infer_sim.core.operators.factories.normalization import NormalizationOpFactory
from llm_infer_sim.core.operators.factories.v4_attention import V4AttentionOpFactory


@dataclass(frozen=True)
class FactoryBundle:
    dense: DenseOpFactory
    norm: NormalizationOpFactory
    embedding: EmbeddingOpFactory
    attention: AttentionOpFactory
    # 3a 起加入: MoE / collective. Dense-only 模型可传 None.
    moe: MoEOpFactory | None = None
    collective: CollectiveOpFactory | None = None
    # 3c 起加入: V3.2 / V4 indexer + V4 attention.
    indexer: IndexerOpFactory | None = None
    v4_attention: V4AttentionOpFactory | None = None


__all__ = [
    "AttentionOpFactory",
    "CollectiveOpFactory",
    "DenseOpFactory",
    "EmbeddingOpFactory",
    "FactoryBundle",
    "IndexerOpFactory",
    "MoEOpFactory",
    "NormalizationOpFactory",
    "V4AttentionOpFactory",
]
