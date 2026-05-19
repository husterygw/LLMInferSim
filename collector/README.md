# LLMInferSim Collector

数据采集模块. 在真实硬件 + vLLM (后续 SGLang/TRT-LLM) 上测算子 latency, 产 JSONL 数据资产.

跟 sim runtime (`llm_infer_sim/`) **完全解耦** — collector 不 import sim 任何模块, 数据生产线独立.

## 用法

```bash
# (规划中) 跑所有 op
python -m collector.cli \
    --frameworks vllm \
    --models qwen3_4b qwen3_30b_a3b \
    --ops gemm attention moe collective \
    --execution-modes eager cudagraph \
    --num-processes 4 \
    --resume

# smoke test
python -m collector.cli \
    --models qwen3_30b_a3b \
    --ops moe \
    --limit 3 \
    --execution-modes eager \
    --num-processes 1
```

## 目录结构

```
collector/
├── __init__.py
├── README.md                    本文档
├── cli.py                       (TODO) 单一入口
├── scheduler.py                 (TODO) 多进程 GPU worker + restart + resume
├── registry.py                  (TODO) (op, framework) → runner 注册表
├── version_resolver.py          (TODO) 按 framework_version 路由
├── schemas.py                   数据 schema (RawRecord / Case / Metrics / ...)
├── harness.py                   (TODO) timing 框架 (warmup + cudagraph + p10/50/90)
├── writer.py                    (TODO) JSONL append (加锁 + fsync)
├── checkpoint.py                (TODO) per-op resume state
├── env_check.py                 (TODO) GPU 锁频 / driver 版本 preflight
├── cases/                       (TODO) 测试 shape 定义 (framework-agnostic)
│   ├── qwen3_4b.py
│   └── qwen3_30b_a3b.py
├── runners/                     (TODO) 单 case 执行体 (framework-specific)
│   ├── vllm_gemm.py
│   ├── vllm_attention.py
│   ├── vllm_moe.py
│   └── vllm_collective.py
├── distributed/                 (TODO) 多 GPU 通信测量 (torchrun)
│   └── run_collective.py
├── importers/                   (TODO) JSONL → OperatorDB schema 转换
│   └── jsonl_to_operator_db.py
├── tests/                       内嵌测试
│   ├── conftest.py
│   ├── test_schemas.py
│   └── ...
└── data/                        数据资产 (.gitignore)
    └── operator_db/<HW>/<framework-version>/
        ├── <op>.jsonl                main 输出
        ├── errors/<op>.jsonl         失败 case
        ├── checkpoints/<op>.json     resume state
        ├── progress.jsonl            跨 op 总进度
        └── manifest.yaml             采集环境快照
```

## 设计原则

1. **跟 sim runtime 零依赖**: collector 内部不 `import llm_infer_sim.core/.adapters`
2. **schemas 是唯一 source of truth**: 所有 read/write 经过 `schemas.py` 定义的 dataclass
3. **失败 case 隔离**: 一个 case 挂不阻塞整批, 写 `errors/<op>.jsonl`
4. **case_id 稳定**: params hash 决定, resume / dedup 都依赖
5. **多 framework 一套设计**: 加 SGLang/TRT-LLM 只需要新 runner + registry entry, scheduler/harness/writer 不动
6. **single-GPU 走主 scheduler, multi-GPU 走 torchrun**: collective 类 case 独立路径

## 测试

```bash
# CPU-only tests (schema / scheduler mock / writer 等)
pytest collector/tests/

# 含 GPU 的 smoke test (需要 vLLM + RTX 4090)
pytest collector/tests/ -m gpu
```

## Schema 演进

`SCHEMA_VERSION = "collector-v1"` 写在每条 record. importer 按版本路由, 老数据保留兼容.

## TODO (实现状态)

- [x] `schemas.py` + 测试
- [ ] `harness.py` + 测试
- [ ] `writer.py` + 测试
- [ ] `checkpoint.py` + 测试
- [ ] `registry.py` + `version_resolver.py` + 测试
- [ ] `scheduler.py` + 测试 (mock crash 重启)
- [ ] `cli.py` (dry-run / list-ops 先跑通)
- [ ] `cases/qwen3_4b.py` + `qwen3_30b_a3b.py`
- [ ] `runners/vllm_gemm.py` (第一个真实 runner)
- [ ] `runners/vllm_attention.py`
- [ ] `runners/vllm_moe.py` (含 routing distribution)
- [ ] `distributed/run_collective.py`
- [ ] `importers/jsonl_to_operator_db.py`
- [ ] `env_check.py`
