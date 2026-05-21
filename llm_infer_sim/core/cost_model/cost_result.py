"""Re-export shim for backward compat.

Step 4 migration: GlobalStepCost / PerRankCost 已搬到 core/cost/legacy.py.
此文件作为 backward compat 入口, 等 Step 4.5 全部消费者迁完后可删.
"""
from llm_infer_sim.core.cost.legacy import GlobalStepCost, PerRankCost

__all__ = ["GlobalStepCost", "PerRankCost"]
