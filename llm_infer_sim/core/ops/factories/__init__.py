"""V3 §6.5 OpFactory — 阶段 1: dense / norm / embedding / attention (minimal GQA)."""
from dataclasses import dataclass

from llm_infer_sim.core.ops.factories.attention import AttentionOpFactory
from llm_infer_sim.core.ops.factories.dense import DenseOpFactory
from llm_infer_sim.core.ops.factories.embedding import EmbeddingOpFactory
from llm_infer_sim.core.ops.factories.normalization import NormalizationOpFactory


@dataclass(frozen=True)
class FactoryBundle:
    """ModelGraphTemplate.build_step 接收的 factory 集合."""
    dense: DenseOpFactory
    norm: NormalizationOpFactory
    embedding: EmbeddingOpFactory
    attention: AttentionOpFactory


__all__ = [
    "DenseOpFactory",
    "NormalizationOpFactory",
    "EmbeddingOpFactory",
    "AttentionOpFactory",
    "FactoryBundle",
]
