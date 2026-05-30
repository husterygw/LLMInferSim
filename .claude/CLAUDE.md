# LLMInferSim

LLM 推理性能模拟器(package `llm-infer-sim`)。**不是独立 server**——它是一个 vLLM
OOT platform plugin(`VirtualPlatform`):跑真实 vLLM 进程,但把 GPU 计算/通信替换成
cost model 估算,从而预测 latency / 吞吐而无需真显卡执行。

入口注册在 `pyproject.toml` 的 `vllm.platform_plugins`:
`virtual = llm_infer_sim.adapters.vllm.virtual_platform:virtual_platform_plugin`

## 环境

- 跑任何 python/pip **先** `conda activate llm_sim`。
- 跑模拟器(含 examples / bench)必加这组 env,缺一个就挂:
  ```bash
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0   # ifwa 的 torch autoload 会破 torch
  export VLLM_VIRTUAL_BACKEND=1            # 激活 VirtualPlatform plugin
  export VLLM_USE_V1=1                     # vLLM v1 引擎
  export HF_HUB_OFFLINE=1                  # 走本地 HF cache,HF id 路径不联网否则会挂
  export TRANSFORMERS_OFFLINE=1
  ```
- 可选:`LLM_INFER_SIM_DUMP_OPS=0/1/2`(per-op breakdown)、`LLM_INFER_SIM_HW=<profile>`、
  `LLM_INFER_SIM_TIME_MODE=realtime|instant`。

## 命令

```bash
# 单元测试(testpaths = tests/ + collector/tests/),都是 pure unit
conda activate llm_sim && pytest

# 最轻量探针:只验证 plugin entry_point 注册 + platform 选中,不起 LLM,~5s
python examples/vllm_virtual/run_platform_selected.py

# 端到端 smoke:真起一次 vLLM 跑 opt-125m
python examples/vllm_virtual/run_opt125m.py

# 对比 bench(真实 vLLM vs 虚拟后端),suite 列表见 scripts/README.md
bash scripts/bench/run_bench_suite.sh single_tp1_roofline
```

`bench_cases.py` 是 benchmark 矩阵的唯一来源,`bench_compare.sh` 只执行它生成的 cases。

## 代码结构

- `llm_infer_sim/core/` — 模拟核心:`operators/` `operator_db/` `operator_schema/`
  `cost/`(cost model)`profiles/`(硬件 profile)`models/` `graph/` `simulation/`
  `workload/` `metrics/` `adapter/`
- `llm_infer_sim/adapters/vllm/` — 接入 vLLM 的边界:`virtual_platform.py`
  `virtual_worker.py` `virtual_model_runner.py` `step_extractor.py` `profile_extractor.py`
- `scripts/` — measure_* 标定脚本 + bench 编排;不是测试
- `examples/` — 各模型/并行配置的手动验证脚本,**不属于自动化测试**
- `configs/calibration/`、`docs/`(CALIBRATION_*、SYSTEM_SOLUTION_V3、REFACTOR_DESIGN 等)

## 工程约定 / 坑

- **大重构后必须起一次端到端 sim smoke**(`run_opt125m.py` 或 bench),光 `pytest`
  全绿不代表 engine init 全链路没断。
- 新 cost 公式落地:必须 dump per-op breakdown(`LLM_INFER_SIM_DUMP_OPS`)跟手算
  1:1 对照,只看宏观 TTFT/吞吐方向不够。
- 本地工作机是 **RTX 4090**,新 fixture/preset 默认硬件用 4090,别默认 H100/A100。
- vLLM v1 会留僵尸子进程,`pkill -f vllm` 杀不到:用 `pgrep VLLM::` + `lsof -i tcp:<port>` 双兜底。
- 短 prefill 的 bench TTFT 有 ±10–50ms 抖动,判定确定性看 step 内部 latency,别拿单次 TTFT outlier 当 sim bug。
- 从 llm-viewer / LLMCompass 复制来的 family-specific 占位代码,接新模型前先审计 + 手算,别直接激活。
- 新增 model/hardware 字段先加到**结构化 profile**(`ModelProfile`/`HardwareProfile`/
  `DeploymentProfile`/`RuntimeProfile`/`CalibrationProfile`),再视需要补 flat read facade;
  扁平 `ModelConfig`/`HardwareConfig` 只是兼容边界,生产装配链走
  `build_roofline_engine_from_scenario(scenario)`,不走扁平 `build_roofline_engine()`。
  `.to_legacy()`/`.from_legacy()` 折返只允许出现在 `profile_extractor`(vLLM ingest)+
  `hardware/registry`(硬件域入口);`tests/contract/test_legacy_profile_guard.py` 会盯住这条线。
