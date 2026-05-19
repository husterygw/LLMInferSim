"""LLMInferSim Collector — 数据采集模块.

跟 sim runtime (`llm_infer_sim/`) 完全解耦, 在 vLLM 等框架上实测真实算子 latency,
落地为 JSONL 数据资产, 给 future MeasuredOperatorDB 喂数据.

设计原则:
  - collector 内部不 import llm_infer_sim.core / llm_infer_sim.adapters
  - 输出 schema 由 collector.schemas 唯一定义
  - 失败 case 隔离 (errors/<op>.jsonl), 不阻塞主流程
  - 多 GPU worker process, 支持 resume / retry

入口: `python -m collector.cli ...`
"""
__version__ = "0.1.0"
