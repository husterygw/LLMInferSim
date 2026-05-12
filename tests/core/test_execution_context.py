"""阶段 4 (4-β): DistributedExecutionContext / PerRankCost / StageExecution 结构测试。

只测数据类的字段与默认值, 不测 cost 公式 (那些已在 test_cost_model / test_mixed_attention 验)。
"""
from llm_infer_sim.core.cost_model.cost_result import GlobalStepCost, PerRankCost
from llm_infer_sim.core.planning.execution_context import (
    DistributedExecutionContext,
    context_from_parallel_config,
)
from llm_infer_sim.core.planning.execution_plan import (
    DistributedExecutionPlan,
    RankPlan,
    StageExecution,
)
from llm_infer_sim.core.profiles.deploy import ParallelConfig


# ------- ExecutionContext -------

def test_context_properties_pass_through_parallel_config():
    ctx = DistributedExecutionContext(
        parallel_config=ParallelConfig(tp_size=4, dp_size=2),
        world_size=8,
    )
    assert ctx.tp_size == 4
    assert ctx.dp_size == 2
    assert ctx.ep_size == 1   # enable_ep=False default
    assert ctx.is_distributed


def test_context_from_parallel_config_helper():
    pc = ParallelConfig(tp_size=2, dp_size=1)
    ctx = context_from_parallel_config(pc)
    assert ctx.world_size == 2
    assert ctx.tp_size == 2
    assert ctx.intra_node_size == 2
    assert ctx.inter_node_count == 1


def test_context_tp1_not_distributed():
    ctx = context_from_parallel_config(ParallelConfig(tp_size=1))
    assert not ctx.is_distributed


# ------- PerRankCost -------

def test_per_rank_cost_compute_time_sum():
    prc = PerRankCost(
        rank_id=0,
        model_core_time=1.0,
        runtime_ops_time=0.5,
        communication_time=0.2,
    )
    assert prc.compute_time == 1.5


def test_global_step_cost_per_rank_costs_default_empty():
    cost = GlobalStepCost(step_id=1)
    assert cost.per_rank_costs == []
    assert cost.critical_rank == 0
    assert cost.rank_imbalance == 0.0


def test_global_step_cost_with_per_rank_costs_populated():
    cost = GlobalStepCost(
        step_id=1,
        total_latency=1.0,
        per_rank_costs=[
            PerRankCost(rank_id=0, total_time=1.0),
            PerRankCost(rank_id=1, total_time=1.0),
        ],
    )
    assert len(cost.per_rank_costs) == 2
    assert cost.per_rank_costs[1].rank_id == 1


# ------- StageExecution / RankPlan / DistributedExecutionPlan additive -------

def test_stage_execution_defaults():
    stage = StageExecution(stage_name="attention")
    assert stage.is_parallel
    assert not stage.has_collective
    assert stage.rank_plans == []


def test_distributed_plan_stages_default_empty_for_back_compat():
    """阶段 3 路径必须依赖 stages=[] 才能 fallback 到 layer_results."""
    plan = DistributedExecutionPlan(step_id=42, world_size=2)
    assert plan.stages == []
    assert plan.execution_context is None
    # 阶段 3 字段仍可用
    assert plan.layer_results == []
    assert plan.attention_override is None


def test_rank_plan_op_lists_default_empty():
    rp = RankPlan(rank_id=3)
    assert rp.model_ops == []
    assert rp.runtime_ops == []
    assert rp.comm_ops == []
