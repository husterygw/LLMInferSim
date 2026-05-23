# LLMInferSim 实施方案

> 状态: current architecture plan  
> 日期: 2026-05-23  
> 目标: 以当前 `Operator + grouped runtime + OperatorDB/Roofline + case-driven benchmark` 为主线, 说明已完成的架构边界、后续瘦身计划和验收方式。

## 0. 当前目标

LLMInferSim 的核心目标是做 LLM serving 的可解释性能仿真:

```text
真实 workload / vLLM step
  -> workload shape
  -> model template 生成 Operator plan
  -> OperatorDB 或 Roofline 估算 cost
  -> StepCostTrace / benchmark report
```

当前工程不再维护旧 `cost_model + VirtualOp + core/ops/factories` 运行路径。理想主路径是:

```text
ModelTemplate
  -> GroupedStepPlan[Operator]
  -> CostRouter
  -> StepCostTrace
```

核心原则:

- 模型由 `Operator` 列表或 grouped operator plan 表达。
- 每个 op 是明确语义类, 例如 `GEMM / Attention / ElementWise / Norm / Embedding / Collective / FusedMoE`。
- `Operator` 负责提供 `signature()` 和 `roofline_spec()`。
- `CostRouter` 在 op 外部决定走 OperatorDB 还是 Roofline。
- Benchmark 使用 case-driven suite, 不在执行器里硬编码 batch / ISL / OSL。

## 1. 目标目录结构

核心目录目标:

```text
llm_infer_sim/core/
  operators/
    base.py              # Operator protocol / RooflineSpec
    context.py           # OperatorContext / ModelBuildContext
    gemm.py
    attention.py
    moe.py
    collective.py
    elementwise.py
    norm.py
    embedding.py

  operator_schema/
    signature.py
    canonical.py
    gemm.py
    attention.py
    moe.py
    collective.py

  operator_db/
    schema.py
    store.py
    importers/
      collector_v2.py
    stores/
      memory.py
      jsonl.py

  models/
    qwen.py
    deepseek.py
    layer_partition.py

  graph/
    grouped_plan.py
    step_shape.py

  cost/
    engine.py
    router.py
    trace.py
    roofline_analyzer.py
    backends/
      roofline.py
      operator_db.py
    roofline/
      communication.py
      kv_transfer.py

  profiles/
  workload/
  simulation/
```

不再作为目标结构保留:

```text
core/cost_model/
core/ops/
core/operators/factories/
core/operators/formulas/
core/operators/ops/
core/operators/routing/
core/operators/schema/
VirtualOp runtime path
LegacyDeployConfig
```

如果上述名字仍在生产代码中出现, 应视为迁移遗留。

## 2. 核心数据流

### 2.1 Workload 到 StepShape

输入来自两类来源:

- vLLM adapter 提供的真实 step 信息。
- core 层测试 / benchmark 构造的 synthetic workload。

统一转为:

```text
StepShape
  step_id
  phase: prefill / decode / mixed
  batch/token shape
  kv/cache shape
  runtime metadata
```

要求:

- `input_len / output_len / batch_size` 属 workload, 不进入 `DeployConfig`。
- `execution_mode / backend / tp / pp / dp / ep` 属 deploy/runtime。
- quantization 字节数来自 model/profile context, 不放进 workload。

### 2.2 ModelTemplate 到 GroupedStepPlan

模型模板负责直接构造 semantic operator:

```text
QwenModelTemplate.build_grouped_step(step)
DeepSeekModelTemplate.build_grouped_step(step)
```

输出:

```text
GroupedStepPlan
  groups: tuple[GroupedOperator]

GroupedOperator
  op: Operator
  count: int
  layer_indices: tuple[int, ...]
```

为什么用 grouped:

- decode / prefill 中大量层结构完全相同。
- per-layer 展开会造成 Python overhead, 影响 realtime sim。
- grouped plan 只估算代表 op 一次, 再乘 `count`。

生产路径应只走 grouped plan。full per-layer plan 只能作为 debug / test helper, 不能作为另一条 runtime 主路径。

### 2.3 Operator

每个 runtime op 必须实现:

```python
name: str
op_kind: str
op_subtype: str
phase: str
layer_idx: int | None
dtype: str

shape -> dict
parallel -> dict
runtime -> dict
signature() -> OperatorSignature
roofline_spec() -> RooflineSpec
```

约束:

- `name` 是短语义名, 如 `qkv_proj`, `attn_add`, `attention`。
- `layer_idx` 负责定位。
- report 层可生成 `display_name = layer{idx}.{name}` 或 `name[count=N]`。
- `name/layer_idx` 不进入 OperatorDB signature。

### 2.4 OperatorDB Signature

OperatorDB key 由 `OperatorSignature` 表达:

```text
op_kind
op_subtype
dtype
shape
parallel
runtime
```

进入 DB 的主要类型:

```text
gemm
attention
moe
collective
```

其他 op:

```text
norm
elementwise
embedding
```

首选走 roofline, 不要求 DB hit。

### 2.5 CostRouter

CostRouter 只负责 cost source policy:

```text
roofline_only
operator_db_first
require_operator_db
```

逻辑:

```text
operator_db_first:
  op.signature() -> OperatorDBBackend.lookup()
  hit  -> measured latency
  miss -> RooflineBackend.estimate(op)

roofline_only:
  RooflineBackend.estimate(op)

require_operator_db:
  hit  -> measured latency
  miss -> error
```

不要把 cost policy 放进 `Operator` 内部。

## 3. OperatorDB 与 Collector 对齐

### 3.1 数据来源

collector 输出:

```text
collector/data/operator_db/<hardware>/<framework-version>/<op_kind>.jsonl
```

每条 RawRecord 转为:

```text
OperatorRecord
  signature
  hardware
  framework
  framework_version
  execution_mode
  kernel_source
  latency_us_p50/p10/p90
  source
```

### 3.2 对齐原则

runtime op 和 collector case 必须生成同一个 `OperatorSignature`。

示例 GEMM 对齐字段:

```text
op_subtype
m / n / k
dtype
tp
framework
framework_version
execution_mode
kernel_source
```

注意:

- `m` 是当前 GEMM 的 token/batch 展开维度, decode 对它非常敏感。
- `execution_mode=eager/cudagraph` 必须进入 runtime key。
- 同 shape 同 kernel 的不同 layer 共享一条实测记录。

### 3.3 Roofline vs 实测报告

算子级报告入口:

```bash
conda run -n llm_sim python scripts/report_operator_roofline_gap.py \
  --db-root collector/data/operator_db \
  --hardware RTX_4090 \
  --framework vllm \
  --framework-version 0.19.1 \
  --op-kind gemm \
  --csv /tmp/gemm_roofline_gap.csv
```

首轮闭环:

- GEMM 完整支持 roofline gap。
- attention / moe / collective 可以先作为 coverage 行。
- 不强行用未验证公式给所有 op 计算 gap。

输出关注:

```text
measured_us_p50
roofline_us
roofline_gap = measured_us_p50 / roofline_us
bottleneck
arithmetic_intensity
```

目标:

- cudagraph GEMM P50 gap 接近 `<= 1.2` 比较合理。
- P90 gap `<= 1.5` 可接受。
- 小 m decode case 要单独分析, 不和大 m prefill 混在一起看。

## 4. Roofline 建模

### 4.1 RooflineSpec

所有 op 的 roofline 输入统一为:

```text
RooflineSpec
  flops
  load_weight
  load_act
  store_act
  load_kv_cache
  store_kv_cache
  op_precision
  op_category
  comm_bytes
  comm_type
```

`RooflineBackend` 做:

```text
op.roofline_spec()
  -> RooflineAnalyzer.analyze()
  -> CostTraceEntry(source=roofline)
```

### 4.2 eager / cudagraph

不要按“大模块”简单加 launch overhead。

更合理的阶段化策略:

```text
初版:
  用 op-level roofline 主体时间。
  cudagraph/eager 通过 runtime key 区分 DB 数据。

后续:
  eager 模式按实际 kernel pipeline 建模 launch overlap。
  cudagraph 模式按 graph replay / piecewise graph 加固定开销。
```

当前建议:

- roofline 主体先保持简单。
- 不要过早模拟每个 eager kernel 的 launch。
- 等 TTFT/TPOT 基线稳定后, 再加 graph replay / pipeline correction。

### 4.3 通信

通信公式放在:

```text
core/cost/roofline/communication.py
```

通信 op 可以通过 `Collective` 表达, 但低层 NCCL 公式可以继续是 roofline primitive, 不必强行塞进每个 op class。

## 5. Benchmark 测试体系

### 5.1 原则

benchmark 已改为 case-driven:

```text
bench_cases.py      # 唯一 case matrix
bench_compare.sh    # 执行 cases
run_bench_suite.sh  # suite 入口
analyze_bench.py    # 汇总
```

不要再在执行器里硬编码:

```text
batch
ISL/OSL
request_rate
TP/EP matrix
```

### 5.2 Suite

语义化 suite:

```text
single_tp1_roofline
batch_tp1_sweep
tp_comm_sweep
tp_batch_sweep
long_context_sweep
moe_tp_sweep
moe_ep_sweep
multi_model_regression
```

兼容映射:

```text
A -> single_tp1_roofline
B -> tp_comm_sweep
C -> batch_tp1_sweep
D -> tp_batch_sweep
E -> multi_model_regression
```

### 5.3 默认测试约束

默认:

```text
prefix_cache = off
chunked_prefill = off
max_num_seqs = None
num_warmups = 1
```

为避免 Qwen3-4B intrinsic max length 触发 vLLM scheduler 校验, suite case 可以显式设置:

```text
max_model_len
max_num_batched_tokens
```

例如:

```text
single_tp1_roofline: max_model_len=8192, max_num_batched_tokens=8192
long_context_sweep:  max_model_len=32768, max_num_batched_tokens 根据 concurrency * input_len 提升
```

### 5.4 运行方式

生成 case:

```bash
python scripts/bench_cases.py --suite single_tp1_roofline --out /tmp/cases.jsonl
```

dry-run:

```bash
bash scripts/run_bench_suite.sh single_tp1_roofline --filter-case '*i128_o128*' --dry-run
```

真实运行:

```bash
bash scripts/run_bench_suite.sh single_tp1_roofline
```

分析:

```bash
python scripts/analyze_bench.py /tmp/llm_infer_sim_bench --suite single_tp1_roofline
```

## 6. 测试策略

### 6.1 普通 CI / 单测

普通 CI 不依赖真实 GPU。

覆盖:

```text
Operator shape/signature/roofline_spec
GroupedStepPlan 构建
CostRouter policy
RooflineBackend
OperatorDB exact hit / fallback
benchmark case generation / dry-run
```

建议命令:

```bash
conda run -n llm_sim pytest tests/core tests/scripts -q
```

### 6.2 算子级真实对比

需要 GPU 和 collector 数据。

目标:

- 验证 collector 与 runtime signature 对齐。
- 看 roofline lower bound 与实测 gap。
- 先 GEMM, 再 attention/MoE/collective。

### 6.3 模型级真实对比

需要真实 vLLM benchmark。

顺序:

```text
single_tp1_roofline
batch_tp1_sweep
tp_comm_sweep
tp_batch_sweep
long_context_sweep
moe_tp_sweep / moe_ep_sweep
```

模型级对比主要看:

```text
TTFT gap
TPOT gap
per-op / grouped trace breakdown
real measured op sum vs model-level benchmark
```

## 7. 后续轻量化计划

当前剩余偏厚点主要来自兼容层和双路径。后续按以下顺序收敛。

### Phase 1: GroupedStepPlan 成为唯一 runtime plan

目标:

- 生产 `StepCostEngine` 只走 `build_grouped_step()`。
- `CostRouter.estimate(StepOpPlan)` 移除或降级为 test-only。
- `StepOpPlan` 不再是 runtime 主抽象。

验收:

```bash
rg "router\.estimate\(" llm_infer_sim tests
rg "StepOpPlan" llm_infer_sim tests
conda run -n llm_sim pytest tests/core/cost tests/core/models
```

### Phase 2: 删除 RooflineOperator legacy wrapper

目标:

- 删除通用 `RooflineOperator`。
- 非标准 op 也要有明确类, 不用通用 formula 容器伪装。
- `KVTransfer` 若只是 PD helper, 放在 cost roofline primitive, 不作为 step runtime op。

验收:

```bash
rg "RooflineOperator|KVTransfer\\(" llm_infer_sim tests
conda run -n llm_sim pytest tests/core/operators tests/core/cost
```

### Phase 3: 清 VirtualOp 命名

目标:

- `virtual_op_to_signature` 改为 `operator_to_signature`。
- `*_virtual_op_to_signature` 改为 `*_operator_to_signature`。
- public export 不再暴露旧名。

验收:

```bash
rg "VirtualOp|virtual_op|_virtual_op_to_signature" llm_infer_sim tests scripts
conda run -n llm_sim pytest tests/core/operator_schema tests/core/operator_db
```

### Phase 4: 精简 CostRouter

目标:

- Router 只表达当前真实策略。
- 不预留尚未实现的 ModuleProfile / CommunicationBackend 复杂优先级。
- collective 未支持时行为明确: skip with metadata 或 raise。

验收:

```bash
conda run -n llm_sim pytest tests/core/cost
```

### Phase 5: benchmark 执行器 Python 化

目标:

- 新增 `scripts/bench_compare.py`。
- `bench_compare.sh` 变成 thin wrapper。
- 删除 shell 中 JSONL -> 控制字符 -> bash field parsing 的逻辑。

验收:

```bash
conda run -n llm_sim pytest tests/scripts/test_benchmark_suites.py
bash scripts/run_bench_suite.sh A --filter-case '*i128_o128*' --dry-run
```

### Phase 6: scripts 分层

目标结构:

```text
scripts/
  bench/
  measure/
  profile/
```

保留根目录 wrapper 一段时间, 但 README 只推荐新路径。

### Phase 7: 模型支持边界收敛

目标:

- 如果 DeepSeek V4 runtime 不支持, 删除或改名 config-only adapter。
- 不让“adapter 还在, runtime 报 removed”的状态长期存在。

验收:

```bash
rg "deepseek_v4|V4" llm_infer_sim tests docs
conda run -n llm_sim pytest tests/adapters tests/core/profiles
```

## 8. 最终验收

全局搜索:

```bash
rg "core\\.cost_model|LegacyDeployConfig|VirtualOp|virtual_op|RooflineOperator|core\\.ops\\." llm_infer_sim tests scripts
```

允许例外:

```text
docs/migration notes
明确 deprecated wrapper 的提示
历史报告
```

核心测试:

```bash
conda run -n llm_sim pytest tests/core tests/adapters tests/scripts -q
```

Benchmark dry-run:

```bash
bash scripts/run_bench_suite.sh single_tp1_roofline --filter-case '*i128_o128*' --dry-run
bash scripts/run_bench_suite.sh batch_tp1_sweep --filter-case '*c4*chat*' --dry-run
```

算子级报告:

```bash
conda run -n llm_sim python scripts/report_operator_roofline_gap.py \
  --db-root collector/data/operator_db \
  --hardware RTX_4090 \
  --framework vllm \
  --framework-version 0.19.1 \
  --op-kind gemm \
  --csv /tmp/gemm_roofline_gap.csv
```
