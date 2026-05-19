"""llm_infer_sim.calibration — Layer 1 op-kernel microbench 校准 (详设 §9.4.2 Plan B).

子模块:
  timings.py  : LayerwiseProfileResults 树 → TimingSample 列表 (DFS + ancestor 匹配)
  catalog.py  : 加载 models/*.yaml, vLLM 类名 + ancestor → canonical op
  shots.py    : Shot grid 定义 (dense / attention / per_sequence 三类)
  batch.py    : Shot → vLLM SchedulerOutput 构造 (B.2 加)
  extension.py: vLLM worker_extension_cls, fire() 包 layerwise_profile (B.2 加)
  engine.py   : vLLM lifecycle (B.3 加)
  runner.py   : 顶层 orchestration (B.3 加)
  csv_io.py   : CSV 读写 (B.3 加)
  fit.py      : CSV → EfficiencyProfile YAML (B.5 加)
  __main__.py : CLI 入口 (B.3 加)

入口模式:
  python -m llm_infer_sim.calibration profile --model ... --hardware ...
  python -m llm_infer_sim.calibration fit --raw ... --out ...
"""

# ---------------------------------------------------------------------------
# _typeshed shim — 必须 在 vllm.profiler import 之前 设, 否则 worker 进程
# 起 layerwise_profile 时挂. vLLM 0.20.2 `vllm/profiler/utils.py` 直接
# `from _typeshed import DataclassInstance`, 但 `_typeshed` 是 typing-only stub,
# runtime 没这个 module. 我们注册一个最小 fake 模块. 这条 shim 在 host (起
# vllm.LLM 之前) 和每个 worker 进程 (load LayerwiseProfileExtension 之前) 都跑.
# 灵感来自 LLMServingSim profiler/__init__.py.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types

if "_typeshed" not in _sys.modules:
    _shim = _types.ModuleType("_typeshed")
    _shim.DataclassInstance = object  # type: ignore[attr-defined]
    _sys.modules["_typeshed"] = _shim
    del _shim
del _sys, _types

from llm_infer_sim.calibration.catalog import Catalog, CatalogEntry
from llm_infer_sim.calibration.shots import (
    Shot,
    DENSE_SHOTS,
    ATTENTION_SHOTS,
    PER_SEQUENCE_SHOTS,
)
from llm_infer_sim.calibration.timings import TimingSample, extract_samples

__all__ = [
    "Catalog",
    "CatalogEntry",
    "Shot",
    "DENSE_SHOTS",
    "ATTENTION_SHOTS",
    "PER_SEQUENCE_SHOTS",
    "TimingSample",
    "extract_samples",
]
