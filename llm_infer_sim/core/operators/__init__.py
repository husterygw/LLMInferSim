"""Semantic operator layer (#158 Step 5 flat).

Each op class in its own module:
  - gemm.py        GEMM
  - norm.py        Norm
  - elementwise.py ElementWise
  - embedding.py   Embedding
  - attention.py   Attention
  - collective.py  Collective + AllReduce/AllGather/ReduceScatter/AllToAll/P2P + make_collective() (deprecated)
  - moe.py         FusedMoE
  - base.py        Operator protocol + RooflineSpec + legacy RooflineOperator / KVTransfer
  - context.py     OperatorContext + ModelBuildContext

每个 op class 的 roofline_spec 公式都内联在该 op 文件 (Attention named constructors,
FusedMoE.routed_experts 等). 不再有独立 formulas/ helper module.
"""

from llm_infer_sim.core.operators.attention import Attention
from llm_infer_sim.core.operators.base import (
    RooflineOperator,
    KVTransfer,
    Operator,
    RooflineSpec,
)
from llm_infer_sim.core.operators.collective import (
    AllGather,
    AllReduce,
    AllToAll,
    Collective,
    P2P,
    ReduceScatter,
    make_collective,
)
from llm_infer_sim.core.operators.context import (
    ModelBuildContext,
    OperatorContext,
    build_model_build_context,
    build_operator_context,
)
from llm_infer_sim.core.operators.elementwise import ElementWise
from llm_infer_sim.core.operators.embedding import Embedding
from llm_infer_sim.core.operators.gemm import GEMM
from llm_infer_sim.core.operators.moe import (
    FusedMoE,
    MoERoutingProfile,
    estimate_distinct_experts,
)
from llm_infer_sim.core.operators.norm import Norm

__all__ = [
    "AllGather",
    "AllReduce",
    "AllToAll",
    "Attention",
    "Collective",
    "ElementWise",
    "P2P",
    "ReduceScatter",
    "Embedding",
    "RooflineOperator",
    "FusedMoE",
    "GEMM",
    "KVTransfer",
    "ModelBuildContext",
    "MoERoutingProfile",
    "Norm",
    "estimate_distinct_experts",
    "Operator",
    "OperatorContext",
    "RooflineSpec",
    "build_model_build_context",
    "build_operator_context",
    "make_collective",
]
