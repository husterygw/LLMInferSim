"""Semantic operator layer (#158 Step 5 flat).

Each op class in its own module:
  - gemm.py        GEMM
  - norm.py        Norm
  - elementwise.py ElementWise
  - embedding.py   Embedding
  - attention.py   Attention (flash/GQA)
  - mla.py         MLAAttention (DeepSeek V3 dense MLA)
  - collective.py  Collective + AllReduce/AllGather/ReduceScatter/AllToAll/P2P
  - moe.py         MoE + MoEDispatch + MoERoutingProfile + build_routed_experts / build_moe_dispatch
  - base.py        Operator protocol + OperatorBase 基类 + RooflineSpec
  - context.py     OperatorContext

每个 op class 的 roofline_spec 公式都内联在该 op 文件 (Attention named constructors,
moe build_routed_experts 等). 不再有独立 formulas/ helper module.
"""

from llm_infer_sim.core.operators.attention import Attention
from llm_infer_sim.core.operators.mla import MLAAttention
from llm_infer_sim.core.operators.base import (
    Operator,
    OperatorBase,
    RooflineSpec,
)
from llm_infer_sim.core.operators.collective import (
    AllGather,
    AllReduce,
    AllToAll,
    Collective,
    P2P,
    ReduceScatter,
)
from llm_infer_sim.core.operators.context import (
    OperatorContext,
    build_operator_context,
    build_operator_context_from_scenario,
)
from llm_infer_sim.core.operators.elementwise import ElementWise
from llm_infer_sim.core.operators.embedding import Embedding
from llm_infer_sim.core.operators.gemm import GEMM
from llm_infer_sim.core.operators.moe import (
    MoE,
    MoEDispatch,
    MoERoutingProfile,
    build_moe_dispatch,
    build_routed_experts,
    estimate_distinct_experts,
)
from llm_infer_sim.core.operators.norm import Norm

__all__ = [
    "AllGather",
    "AllReduce",
    "AllToAll",
    "Attention",
    "MLAAttention",
    "Collective",
    "ElementWise",
    "P2P",
    "ReduceScatter",
    "Embedding",
    "GEMM",
    "MoE",
    "MoEDispatch",
    "MoERoutingProfile",
    "Norm",
    "build_moe_dispatch",
    "build_routed_experts",
    "estimate_distinct_experts",
    "Operator",
    "OperatorBase",
    "OperatorContext",
    "RooflineSpec",
    "build_operator_context",
    "build_operator_context_from_scenario",
]
