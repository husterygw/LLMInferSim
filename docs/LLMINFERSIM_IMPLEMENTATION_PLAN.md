# LLMInferSim 实施方案

> 状态: current architecture plan  
> 日期: 2026-05-23  
> 目标: 以当前 `Operator + grouped runtime + OperatorDB/Roofline + case-driven benchmark` 为主线, 说明最终交付能力、功能数据流、模块实施边界和后续瘦身验收方式。

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

## 1. 交付能力与实施切分

本实施方案按系统方案中的五类最终能力来组织。读者可以先从“这个工程能做什么”理解目标, 再看后面的目录结构和核心抽象。

### 1.1 能力一: 模型级性能仿真

目标:

```text
在真实 vLLM scheduler / request lifecycle 中替换模型执行耗时,
输出 TTFT / TPOT / E2E / throughput / step breakdown。
```

实施数据流:

```text
vLLM step context
  -> adapters/vllm 提取 workload
  -> GlobalStepWorkload / StepShape
  -> QwenModelTemplate / DeepSeekModelTemplate
  -> GroupedStepPlan[Operator]
  -> StepCostEngine + CostRouter
  -> StepCostTrace
  -> VirtualModelRunner 推进虚拟时间 / fake output
  -> vLLM scheduler 继续运行
```

落地模块:

```text
llm_infer_sim/adapters/vllm/
llm_infer_sim/core/workload/
llm_infer_sim/core/graph/grouped_plan.py
llm_infer_sim/core/models/
llm_infer_sim/core/cost/engine.py
llm_infer_sim/core/cost/router.py
llm_infer_sim/core/cost/trace.py
```

实施要求:

- production runtime 以 `GroupedStepPlan` 为主, 避免每 step 展开 `num_layers * ops`。
- `full per-layer plan` 只能作为 debug / test helper, 不能再成为另一条主路径。
- `StepCostTrace` 必须能表达 grouped entry 的 `count / layer_indices / source / bottleneck`。
- vLLM worker 层软件开销不应被 Python per-op simulation overhead 污染。

验收:

```bash
conda run -n llm_sim pytest tests/core/cost tests/core/models tests/adapters -q
bash scripts/run_bench_suite.sh single_tp1_roofline --filter-case '*i128_o128*' --dry-run
```

### 1.2 能力二: 算子级 roofline vs 实测对比

目标:

```text
读取 collector OperatorDB, 对齐 runtime Operator signature,
输出每个 shape 的实测 latency、roofline lower bound 和 gap。
```

实施数据流:

```text
collector/data/operator_db/.../*.jsonl
  -> collector_v2 importer
  -> OperatorRecord
  -> OperatorSignature
  -> report_operator_roofline_gap.py
       -> signature 还原 Operator
       -> op.roofline_spec()
       -> RooflineBackend.estimate()
  -> CSV / JSONL / summary
```

落地模块:

```text
scripts/report_operator_roofline_gap.py
llm_infer_sim/core/operator_db/
llm_infer_sim/core/operator_schema/
llm_infer_sim/core/operators/
llm_infer_sim/core/cost/backends/roofline.py
```

实施要求:

- GEMM 是首轮完整闭环: `RawRecord -> GEMM -> signature / roofline_spec -> gap`。
- attention / moe / collective 先保证 coverage 和 signature 对齐, 公式成熟后再纳入 gap。
- `execution_mode / framework_version / kernel_source` 必须进入 signature runtime key。
- `status` 字段若已有外部消费, 修改时要同步历史报告和分析脚本。

近期修正点:

- attention prefill 的 causal flops 应考虑三角形 token 对, 不应简单按 `S * S`。
- FlashAttention KV I/O 要区分 causal 平均访问量与 tile 重读。
- attention 输出 O 的 read/write 要显式表达, 避免在 `store_act` 中含混。
- `kv_dtype` 不应长期硬编码为 bf16, 应来自 model/deploy/profile context。

验收:

```bash
conda run -n llm_sim pytest tests/core/operator_schema tests/core/operator_db tests/scripts/test_report_operator_roofline_gap.py -q
conda run -n llm_sim python scripts/report_operator_roofline_gap.py \
  --db-root collector/data/operator_db \
  --hardware RTX_4090 \
  --framework vllm \
  --framework-version 0.19.1 \
  --op-kind gemm \
  --csv /tmp/gemm_roofline_gap.csv
```

### 1.3 能力三: 模型级 real-vs-sim benchmark

目标:

```text
同一组 benchmark case 分别跑真实 vLLM 和 sim backend,
输出 TTFT / TPOT / E2E / throughput gap。
```

实施数据流:

```text
bench_cases.py
  -> cases.jsonl
  -> bench_compare executor
       -> real vLLM server + vllm bench serve
       -> sim vLLM server + vllm bench serve
  -> raw result
  -> _extract_metrics.py
  -> analyze_bench.py
```

落地模块:

```text
scripts/bench_cases.py
scripts/bench_compare.sh
scripts/run_bench_suite.sh
scripts/analyze_bench.py
scripts/_extract_metrics.py
```

实施要求:

- `bench_cases.py` 是唯一 case matrix 来源。
- `bench_compare` 只负责执行, 不再内置 batch / ISL / OSL / TP matrix。
- suite 名必须语义化, 不再以 Stage A/B/C/D 作为主文档入口。
- 默认不开 prefix cache, 不开 chunked prefill。
- `max_model_len / max_num_batched_tokens` 由 case 显式给出, 解决 vLLM 启动校验。

验收:

```bash
conda run -n llm_sim pytest tests/scripts/test_benchmark_suites.py -q
bash scripts/run_bench_suite.sh single_tp1_roofline --filter-case '*i128_o128*' --dry-run
python scripts/analyze_bench.py /tmp/llm_infer_sim_bench --suite single_tp1_roofline
```

### 1.4 能力四: 调试与归因

目标:

```text
当模型级 real-vs-sim 有 gap 时, 能拆到 phase / op kind / op subtype / shape / source。
```

实施数据流:

```text
model benchmark gap
  -> StepCostTrace
  -> grouped operator breakdown
  -> OperatorDB exact hit / roofline fallback 标记
  -> operator roofline gap report
  -> 定位是算子公式、DB coverage、framework runtime overhead 还是 benchmark variance
```

落地模块:

```text
llm_infer_sim/core/cost/trace.py
scripts/analyze_bench.py
scripts/report_operator_roofline_gap.py
后续: scripts/debug_breakdown.py
```

实施要求:

- trace entry 必须保留 `op_kind / op_subtype / shape / source / count`。
- grouped trace 要能展开到“这一类 op 代表了哪些 layer”。
- 支持把模型级某个 case 的 shape 映射回 collector 中同 shape 或近邻 shape 的记录。
- 不把 vLLM scheduler 开销、worker runtime overhead、kernel roofline gap 混为一个数字。

验收:

```text
给定 single_tp1_roofline 中一个 case,
能输出按 op_subtype 聚合的 sim cost,
并列出 GEMM 类对应 collector measured latency / roofline latency / gap。
```

### 1.5 能力五: 部署配置搜索准备

目标:

```text
先把单次仿真的 deploy/workload/model 边界收干净,
后续再在外层枚举 TP / PP / DP / EP / execution_mode / max_num_seqs 等候选。
```

实施数据流:

```text
candidate DeployConfig
  + ModelConfig
  + Workload / BenchmarkCase
  -> one simulation run
  -> metrics + trace
  -> search policy 汇总候选
```

实施要求:

- `DeployConfig` 保持最小集, 不放 workload 和 quantization 字段。
- `input_len / output_len / concurrency` 属 workload / benchmark case。
- `w_byte / a_byte / kv_byte` 属 model/profile/quantization context。
- 配置搜索暂不作为近期主任务, 但当前目录和数据结构不能阻碍后续接入。

验收:

```text
同一 workload 仅替换 DeployConfig,
生成的 Operator parallel/runtime metadata 随之变化,
CostRouter 能用同一套路径完成估算。
```

## 2. 目标目录结构

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

## 3. 核心实现数据流

### 3.1 Workload 到 StepShape

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

### 3.2 ModelTemplate 到 GroupedStepPlan

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

### 3.3 Operator

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

### 3.4 OperatorDB Signature

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

### 3.5 CostRouter

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

## 4. OperatorDB 与 Collector 对齐

### 4.1 数据来源

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

### 4.2 对齐原则

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

### 4.3 Roofline vs 实测报告

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

## 5. Roofline 建模

### 5.1 RooflineSpec

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

### 5.2 eager / cudagraph

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

### 5.3 通信

通信公式放在:

```text
core/cost/roofline/communication.py
```

通信 op 可以通过 `Collective` 表达, 但低层 NCCL 公式可以继续是 roofline primitive, 不必强行塞进每个 op class。

## 6. Benchmark 测试体系

### 6.1 原则

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

### 6.2 Suite

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

### 6.3 默认测试约束

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

### 6.4 运行方式

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

## 7. 测试策略

### 7.1 普通 CI / 单测

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

### 7.2 算子级真实对比

需要 GPU 和 collector 数据。

目标:

- 验证 collector 与 runtime signature 对齐。
- 看 roofline lower bound 与实测 gap。
- 先 GEMM, 再 attention/MoE/collective。

### 7.3 模型级真实对比

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

## 8. 工程总实施计划

这一节不是“后续轻量化”, 而是整个 LLMInferSim 工程从当前状态走向最终系统形态的实施路线。每个阶段都对应前面的一类能力或一组基础设施。

当前总体进度:

```text
阶段 0  文档与目标架构收敛                 done
阶段 1  Operator 新核心骨架                 mostly done
阶段 2  Qwen dense grouped runtime           mostly done
阶段 3  OperatorDB / GEMM roofline 闭环       mostly done
阶段 4  算子公式校准与模型级归因             in progress
阶段 5  Benchmark 体系稳定化                 mostly done
阶段 6  通信 / TP / EP / MoE 校准             not started / partial
阶段 7  DeepSeek / MLA / 非标准 op 收敛       partial
阶段 8  旧兼容层清理                         in progress
阶段 9  配置搜索                             not started
```

按当前代码扫描, 工程已经不再是旧 `core.cost_model` 主路径, `StepCostEngine` 已经走 `build_grouped_step()`。但仍有几类遗留:

- `StepOpPlan` 还在 `CostRouter.estimate()` 和部分测试中, 需要降级为 test/debug helper 或删除。
- `RooflineOperator / KVTransfer` 还在 DeepSeek 非标准 op 和测试中, 需要替换为明确 op 或 cost roofline primitive。
- `virtual_op_to_signature` 这类旧命名还在 operator schema 和测试中, 需要统一改成 `operator_to_signature`。
- benchmark 已是 case-driven, 但执行器仍是 shell 主体, 后续可 Python 化。

### Phase 0: 系统方案与实施方案收敛

状态: done

目标:

- 系统方案明确最终能力、功能数据流和核心抽象。
- 实施方案按能力拆分, 能看出每个能力落到哪些模块。
- 不再用旧 V3 draft 的 `VirtualOp + OpFactory + StepOpPlan` 作为最终主线。

已完成:

- `Operator + GroupedStepPlan + CostRouter` 确认为主路径。
- 系统方案加入“最终能力与功能数据流”。
- 实施方案加入“交付能力与实施切分”。

验收:

```bash
rg "最终能力|交付能力|GroupedStepPlan|OperatorDB" docs/LLMINFERSIM_SYSTEM_SOLUTION_V3.md docs/LLMINFERSIM_IMPLEMENTATION_PLAN.md
```

### Phase 1: Operator 核心骨架

状态: mostly done

目标:

- 用 semantic operator 表达模型结构。
- `GEMM / Attention / ElementWise / Norm / Embedding / Collective / FusedMoE` 成为主类。
- `Operator` 提供 `signature()` 和 `roofline_spec()`。
- cost policy 不放进 op 内部。

已完成:

- Qwen 主路径已能构造 semantic operator。
- GEMM 已能从 collector record 还原并做 roofline report。
- `OperatorContext / ModelBuildContext` 已承接 dtype、byte、runtime 等上下文。

剩余:

- `RooflineOperator / KVTransfer` 仍是 legacy wrapper。
- DeepSeek 非标准 op 还没有完全 semantic 化。
- 部分测试仍通过 `RooflineOperator` 构造 wrong-kind case。

验收:

```bash
rg "RooflineOperator|KVTransfer\\(" llm_infer_sim tests
conda run -n llm_sim pytest tests/core/operators tests/core/models -q
```

### Phase 2: Grouped runtime 成为唯一生产路径

状态: mostly done

目标:

- 生产 `StepCostEngine` 只走 `build_grouped_step()`。
- 避免每 step 展开 `num_layers * ops` 带来的 Python overhead。
- full per-layer plan 只用于 debug / 单测, 不进入 vLLM worker 主路径。

已完成:

- Qwen grouped path 已存在。
- `StepCostEngine` 已使用 grouped plan。
- `CostRouter.estimate_grouped()` 已能聚合 `count * op_latency`。

剩余:

- `StepOpPlan` 文件和 `CostRouter.estimate(StepOpPlan)` 仍存在。
- 部分 cost 测试仍基于 `StepOpPlan`。
- 需要补一个明确的 debug-only 边界, 或直接删除 full plan。

验收:

```bash
rg "StepOpPlan|router\\.estimate\\(" llm_infer_sim tests
conda run -n llm_sim pytest tests/core/cost tests/core/models -q
```

### Phase 3: OperatorDB 与 GEMM roofline 闭环

状态: mostly done

目标:

- collector 数据和 runtime operator 通过同一套 `OperatorSignature` 对齐。
- GEMM report 输出 measured / roofline / gap。
- DB exact hit 和 roofline fallback 都可测。

已完成:

- `scripts/report_operator_roofline_gap.py` 已有 GEMM 主线。
- collector v2 importer 已转成 `OperatorRecord`。
- benchmark 与 collector 的 `execution_mode / kernel_source / framework_version` 已进入 key。

剩余:

- `virtual_op_to_signature` 命名仍未清理。
- attention / moe / collective 还主要是 coverage, 未形成稳定 gap 校准。

验收:

```bash
rg "virtual_op|_virtual_op_to_signature" llm_infer_sim tests scripts
conda run -n llm_sim pytest tests/core/operator_schema tests/core/operator_db tests/scripts/test_report_operator_roofline_gap.py -q
```

### Phase 4: 算子公式校准与模型级归因

状态: in progress

目标:

- 用算子级实测解释模型级 TTFT/TPOT gap。
- 修正 attention / MoE / communication 的 roofline spec。
- 能把一个 benchmark case 拆成 grouped op breakdown, 再映射到 collector shape。

当前重点:

- attention prefill 的 causal flops 不能简单按 `S * S`。
- FlashAttention KV I/O 要区分 causal 半三角访问与 tile 重读。
- attention O 的 read/write 要显式建模。
- `kv_dtype` 要来自 context, 不能长期硬编码。
- cudagraph/piecewise graph overhead 暂不混入 op roofline, 后续作为 runtime correction。

验收:

```bash
conda run -n llm_sim pytest tests/core/operators tests/core/cost -q
conda run -n llm_sim python scripts/report_operator_roofline_gap.py \
  --db-root collector/data/operator_db \
  --hardware RTX_4090 \
  --framework vllm \
  --framework-version 0.19.1 \
  --op-kind gemm \
  --csv /tmp/gemm_roofline_gap.csv
```

阶段完成标志:

```text
给定一个 single_tp1_roofline case,
能列出每类 GEMM / attention / norm / elementwise 的 sim cost,
并说明哪些来自 OperatorDB、哪些来自 roofline、哪些缺少 collector coverage。
```

### Phase 5: Benchmark 体系稳定化

状态: mostly done

目标:

- `bench_cases.py` 成为唯一 benchmark matrix。
- `bench_compare` 只执行 cases。
- suite 名语义化。
- 默认关闭 prefix cache / chunked prefill。

已完成:

- 新 suite 已定义:

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

- A/B/C/D/E 已变成兼容 alias。
- case 中已显式处理 `max_model_len / max_num_batched_tokens`。

剩余:

- `bench_compare.sh` 仍是 shell 执行器, 内部有嵌入式 Python。
- scripts 目录仍比较拥挤。

验收:

```bash
conda run -n llm_sim pytest tests/scripts/test_benchmark_suites.py -q
bash scripts/run_bench_suite.sh single_tp1_roofline --filter-case '*i128_o128*' --dry-run
```

### Phase 6: 通信 / TP / EP / MoE 校准

状态: not started / partial

目标:

- TP allreduce / allgather / reducescatter / p2p 能在 TP suite 中解释 gap。
- MoE routed expert compute、dispatch/combine、routing distribution 能在 MoE suite 中拆解。
- EP all-to-all 数据量与 collector collective 数据对齐。

实施内容:

- `Collective` signature 与 collector collective case 对齐。
- 通信 roofline primitive 补 topology / protocol / algorithm 字段。
- MoE routing profile 输出每 rank token count、active expert count、all-to-all bytes。
- Qwen3-30B-A3B TP-only 和 EP suite 加入真实校准。

验收:

```bash
bash scripts/run_bench_suite.sh tp_comm_sweep --dry-run
bash scripts/run_bench_suite.sh moe_ep_sweep --dry-run
conda run -n llm_sim pytest tests/core/test_moe_cost_consistency.py tests/core/test_ep_cost_consistency.py -q
```

### Phase 7: DeepSeek / MLA / 非标准 op 收敛

状态: partial

目标:

- DeepSeek 模型构造遵循同一套 `Operator + GroupedStepPlan`。
- MLA / indexer / fused_compress 等非标准 op 不再依赖通用 `RooflineOperator`。
- 如果 DeepSeek V4 runtime 暂不支持, 文档和代码边界要明确, 不保留半工作路径。

实施内容:

- 为 indexer fp8 GEMM、fused compress、MLA attention 建明确 operator 或明确 cost primitive。
- DeepSeek V4 若要重做, 按 AIConfigurator V4 结构重新建模, 不迁就旧实现。
- 清理 adapter/config-only 与 runtime 支持状态不一致的问题。

验收:

```bash
rg "RooflineOperator|deepseek_v4|V4" llm_infer_sim tests docs
conda run -n llm_sim pytest tests/core/test_mla_cost_consistency.py tests/adapters tests/core/profiles -q
```

### Phase 8: 旧兼容层清理

状态: in progress

目标:

- 删除或改名所有旧 runtime 术语。
- 生产代码不再出现 `core.cost_model / LegacyDeployConfig / VirtualOp / core.ops`。
- schema API 从 `virtual_op_to_signature` 收敛为 `operator_to_signature`。

实施顺序:

```text
1. 清 operator_schema 旧命名
2. 清 StepOpPlan runtime 入口
3. 清 RooflineOperator / KVTransfer legacy wrapper
4. 清 core/ops / cost_model / LegacyDeployConfig 残留
5. 更新测试与文档
```

验收:

```bash
rg "core\\.cost_model|LegacyDeployConfig|VirtualOp|virtual_op|RooflineOperator|core\\.ops\\." llm_infer_sim tests scripts
conda run -n llm_sim pytest tests/core tests/adapters tests/scripts -q
```

### Phase 9: 配置搜索

状态: not started

目标:

- 在 core 层实现不依赖 vLLM sleep 的快速部署配置搜索。
- 枚举 TP / PP / DP / EP / execution_mode / max_num_seqs / memory budget 等候选。
- 输出 Pareto frontier 和每个候选的瓶颈解释。

实施边界:

- 不在近期打断 roofline 校准主线。
- 当前阶段只保证 `DeployConfig / Workload / ModelConfig` 边界足够干净。
- 搜索器作为外层能力调用已有 simulation engine, 不反向污染 core operator 设计。

验收:

```text
给定同一 workload,
搜索器能枚举多个 DeployConfig,
复用同一套 GroupedStepPlan / CostRouter 路径得到指标,
并输出每个候选的 TTFT / TPOT / memory / bottleneck。
```

## 9. 最终验收

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
