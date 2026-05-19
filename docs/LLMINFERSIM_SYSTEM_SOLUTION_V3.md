# LLMInferSim 完整系统方案

> 状态: v3 draft  
> 日期: 2026-05-19  
> 输入来源: `inference_sim_system_solution_v2.md` + `OPERATOR_DB_REFACTOR_DRAFT.md`  
> 目标: 给出一份可讨论的完整系统方案。后续再拆成具体实施设计与任务。

## 1. 系统定位

LLMInferSim 应定位为:

```text
vLLM-native inference performance simulator
  = 真实 vLLM scheduler
  + framework-independent workload IR
  + op graph / VirtualOp
  + OperatorDB / ModuleProfile / Roofline 多级 cost backend
  + virtual execution / fake output
  + request-level metrics and configuration search
```

它不是 AIConfigurator 的简单复刻, 也不是 LLMServingSim 的外部 scheduler simulator。

核心差异是:

```text
保留真实推理框架的 scheduler 和请求生命周期,
只替换 model execution 的耗时与输出。
```

因此本系统的主路径应是:

```text
Framework Adapter
  -> GlobalStepWorkload
  -> StepShape
  -> ModelGraphTemplate
  -> StepOpPlan
  -> CostRouter
  -> StepCostTrace
  -> Virtual Time / Fake Output
  -> Framework Scheduler Continues
```

## 2. 参考系统吸收边界

### 2.1 从 AIConfigurator 吸收

AIConfigurator 最值得借鉴:

- 模型虚拟 op 构建: `Model -> Operation list`
- typed operator database
- operator collector / microbench
- kernel_source / framework_version / graph/eager 数据管理
- MoE power-law routing correction
- 配置搜索、Pareto frontier、SLA picking

不照搬:

- steady-state serving 估计主路径
- 与真实 scheduler 脱离的 batch 假设
- 模型/框架强绑定的数据组织方式

### 2.2 从 LLMServingSim 吸收

LLMServingSim 最值得借鉴:

- 请求级 workload 和 metrics
- attention mixed-batch shape
- module profile
- communication event 与拓扑解耦
- virtual time / fast simulation 思路

不照搬:

- 在 vLLM 外部维护完整 scheduler 近似
- 只依赖 module profile 而缺少底层 OperatorDB 的路线

### 2.3 本系统主张

本系统把两者能力放进真实框架执行链路:

```text
真实 vLLM scheduler
  + AIConfigurator-style OperatorDB/Collector
  + LLMServingSim-style request metrics/module profile
  + LLMInferSim op graph and cost router
```

## 3. 总体架构

系统分为两条平面:

```text
在线仿真执行平面
离线数据采集与校准平面
```

### 3.1 在线仿真执行平面

```text
┌─────────────────────────────────────────────────────────────┐
│ 推理框架层                                                   │
│ vLLM / future SGLang                                         │
│ request input / scheduler / KV cache / output processor      │
└──────────────────────────────┬──────────────────────────────┘
                               │ framework step context
┌──────────────────────────────▼──────────────────────────────┐
│ Adapter 层                                                   │
│ VllmAdapter / StepExtractor / VirtualWorker / VirtualRunner  │
│ 框架对象 -> GlobalStepWorkload, StepCostTrace -> fake output │
└──────────────────────────────┬──────────────────────────────┘
                               │ GlobalStepWorkload
┌──────────────────────────────▼──────────────────────────────┐
│ Workload / Shape 层                                          │
│ GlobalStepWorkload / RequestWorkload / StepShape             │
│ 表示本 step 的真实 batch、token、context、graph mode          │
└──────────────────────────────┬──────────────────────────────┘
                               │ StepShape
┌──────────────────────────────▼──────────────────────────────┐
│ Op Graph 层                                                  │
│ ModelGraphTemplate / OpFactory / RoutingProfile              │
│ StepShape + ModelConfig -> StepOpPlan(VirtualOp list)        │
└──────────────────────────────┬──────────────────────────────┘
                               │ StepOpPlan
┌──────────────────────────────▼──────────────────────────────┐
│ Cost 层                                                      │
│ CostRouter / ModuleProfile / OperatorDB / Formula / Roofline │
│ 对每个 VirtualOp 输出 latency、source、confidence、gap        │
└──────────────────────────────┬──────────────────────────────┘
                               │ StepCostTrace
┌──────────────────────────────▼──────────────────────────────┐
│ Simulation / Metrics 层                                      │
│ VirtualClock / SleepExecutor / FakeOutput / Reporter         │
│ 推进虚拟时间, 输出 TTFT/TPOT/E2E/throughput/breakdown         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 离线数据采集与校准平面

```text
┌─────────────────────────────────────────────────────────────┐
│ Collector / Profiler                                        │
│ OperatorMicrobench / ModuleMicrobench / StepProfile / E2E   │
│ 在真实硬件或 proxy 硬件上采集 raw data                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ RawRecord JSONL / traces
┌──────────────────────────────▼──────────────────────────────┐
│ Normalizer / Importer                                       │
│ collector_v2 importer / schema validator / canonicalizer    │
│ 将 raw data 规范化为 OperatorRecord / ModuleProfile          │
└──────────────────────────────┬──────────────────────────────┘
                               │ normalized profile data
┌──────────────────────────────▼──────────────────────────────┐
│ 数据资产层                                                   │
│ OperatorDB / ModuleProfileBundle / CommDB / DerivedProfile   │
│ 保存实测数据、派生 profile、proxy/envelope 数据               │
└──────────────────────────────┬──────────────────────────────┘
                               │ queryable artifacts
┌──────────────────────────────▼──────────────────────────────┐
│ Calibration / Regression                                    │
│ interpolation / scaling / envelope / confidence / validation │
│ 生成 CostBackend 可消费的数据与策略                          │
└─────────────────────────────────────────────────────────────┘
```

## 4. 核心抽象

### 4.1 GlobalStepWorkload

`GlobalStepWorkload` 是 adapter 输出的 framework-independent step workload。

它描述:

- step phase: prefill / decode / mixed / chunked_prefill
- request list
- prefill/decode token 数
- request context length
- prefix cache 命中
- chunked prefill 状态

它保留为框架适配层和 core 的边界对象。

### 4.2 StepShape

`StepShape` 是 graph/cost 层消费的 step shape。

它从 `GlobalStepWorkload` 派生, 加入 graph/cudagraph 和 cost 相关字段:

```python
@dataclass(frozen=True)
class StepShape:
    step_id: int
    phase: str
    total_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    num_prefill_requests: int
    num_decode_requests: int
    max_context_len: int
    max_prefill_seqlen: int
    avg_decode_context_len: int
    execution_mode: str
    graph_capture_size: int | None = None
    padded_tokens: int | None = None
```

### 4.3 VirtualOp

`VirtualOp` 是新架构中心对象。

它是:

```text
可查询的 op descriptor
  + roofline fallback feature
  + trace unit
```

```python
@dataclass(frozen=True)
class VirtualOp:
    name: str
    op_kind: str
    op_subtype: str
    phase: str
    layer_idx: int | None

    dtype: str
    shape: dict[str, Any]
    parallel: dict[str, Any]
    runtime: dict[str, Any]

    formula: dict[str, Any]
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
```

其中:

- `op_kind/op_subtype/dtype/shape/parallel/runtime` 用于 OperatorDB query。
- `formula` 用于 roofline / communication formula fallback。
- `name/layer_idx/phase/tags` 用于 trace 和报告。

### 4.4 StepOpPlan

`StepOpPlan` 是一个 step 的 op graph。

首版可以是 flat list:

```python
@dataclass(frozen=True)
class StepOpPlan:
    step_id: int
    phase: str
    ops: tuple[VirtualOp, ...]
    metadata: dict[str, Any]
```

后续扩展为:

```text
StepOpPlan
  StagePlan[]
    RankPlan[]
      VirtualOp[]
```

用于 TP/EP/DP、collective、overlap、critical path 和 asymmetric MoE。

### 4.5 StepCostTrace

Cost 层输出不应该只是一个总 latency, 而应是可解释 trace:

```python
@dataclass(frozen=True)
class CostTraceEntry:
    op_name: str
    op_kind: str
    op_subtype: str
    latency_s: float
    source: str          # module_profile / operator_db / comm_formula / roofline
    match_type: str      # exact / nearest / fallback
    roofline_s: float | None
    roofline_gap: float | None
    metadata: dict[str, Any]
```

```python
@dataclass(frozen=True)
class StepCostTrace:
    step_id: int
    phase: str
    total_latency_s: float
    compute_time_s: float
    memory_time_s: float
    comm_time_s: float
    runtime_time_s: float
    entries: tuple[CostTraceEntry, ...]
    bottleneck: str
```

### 4.6 DeployConfig

`DeployConfig` 是单次仿真的部署配置对象, 也是后续最佳部署搜索的参数边界。

当前阶段不实现搜索, 但所有 core 入口都必须能被 `DeployConfig` 参数化:

```python
@dataclass(frozen=True)
class DeployConfig:
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    moe_tp_size: int = 1
    moe_ep_size: int = 1

    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    block_size: int = 16
    num_gpu_blocks: int | None = None

    execution_mode: str = "eager"
    backend: str = "vllm"
    backend_version: str | None = None
```

它的作用不是枚举候选, 而是保证单次仿真 search-ready:

- `StepShape` 从中读取 execution mode、block size、graph capture 等运行时信息。
- `OpFactory` 从中读取 TP/EP/DP/MoE parallel 信息。
- `VirtualOp.parallel/runtime` 携带这些字段, 作为 OperatorDB query 的一部分。
- `CostRouter` 和 `StepCostTrace` 保留 deploy metadata, 便于后续多配置结果合并。

后期搜索应只是外层循环:

```python
for candidate in candidates:
    result = simulator.run(workload, model_config, candidate.deploy_config)
```

如果后期为了搜索还需要重写 `ops`、`operator_db` 或 `cost` 的核心边界,
说明当前 `DeployConfig` 约束没有落实好。

## 5. Operator Schema Contract

Operator Schema Contract 是 collector、runtime、OperatorDB 之间的唯一契约。

硬规则:

```text
collector Case.params
runtime VirtualOp.shape/parallel/runtime
operator_db OperatorRecord.signature

必须 canonicalize 成同一个 OperatorSignature。
```

### 5.1 GEMM

统一字段:

```text
op_kind = gemm
op_subtype = qkv_proj / o_proj / gate_up_proj / down_proj / lm_head / router / ...
dtype
shape = {m, n, k}
parallel = {tp}
runtime = {framework, framework_version, execution_mode, kernel_source}
```

collector case 示例:

```python
{
    "op_subtype": "qkv_proj",
    "m": 128,
    "n": 6144,
    "k": 2560,
    "dtype": "bf16",
    "tp": 1,
}
```

runtime `VirtualOp` 必须生成同构字段。

### 5.2 MoE

统一字段:

```text
op_kind = moe
op_subtype = fused_moe
dtype
shape = {
  num_tokens,
  hidden,
  moe_intermediate,
  topk,
  num_experts,
  routing_distribution,
  power_law_alpha
}
parallel = {tp, ep}
runtime = {framework, framework_version, execution_mode, kernel_source}
```

`routing_distribution` 必须进入 key。balanced 与 power_law 不能互相命中。

### 5.3 Attention

首版统一字段:

```text
op_kind = attention
op_subtype = prefill / decode / mixed_split / mixed_unified
dtype
shape = {
  num_tokens,
  num_seqs,
  q_len,
  kv_len,
  num_q_heads,
  num_kv_heads,
  head_dim
}
parallel = {tp}
runtime = {
  attention_backend,
  kv_dtype,
  block_size,
  execution_mode,
  kernel_source
}
```

后续扩展 MLA、sparse attention、ragged batch、graph capture size。

### 5.4 Collective

统一字段:

```text
op_kind = collective
op_subtype = allreduce / allgather / reduce_scatter / alltoall / p2p
dtype
shape = {message_bytes}
parallel = {world_size, tp, ep, node_count, gpus_per_node}
runtime = {backend=nccl, algo, protocol, topology, execution_mode}
```

## 6. ModelGraphTemplate 与 OpFactory

### 6.1 ModelGraphTemplate

模型模板只描述结构, 不算 latency, 不查 DB。

职责:

- 判断每层是 dense 还是 MoE。
- 判断 attention 类型: MHA/GQA/MLA/V4 sparse。
- 判断是否有 shared expert、hash routing、hyper-connection。
- 根据 `StepShape` 和 factories 生成 `VirtualOp`。

示例:

```python
class ModelGraphTemplate:
    def build_step(self, step: StepShape, factories: FactoryBundle) -> StepOpPlan:
        ...
```

### 6.2 Qwen Dense Layer

生成:

```text
attn_norm
qkv_proj
rope
attention
o_proj
attn_allreduce?
attn_add
mlp_norm
gate_up_proj
mlp_act
down_proj
mlp_allreduce?
mlp_add
```

### 6.3 Qwen MoE Layer

生成:

```text
attn block
mlp_norm
router
moe_dispatch?
fused_moe
moe_combine?
shared_expert?
mlp_add
```

### 6.4 DeepSeek Template

负责表达:

- MLA
- shared experts
- V3/V4 sparse attention
- hash routing
- hyper-connection
- FP4/FP8 expert precision hints

但仍然只生成 `VirtualOp`。

### 6.5 OpFactory 分类

```text
DenseOpFactory
  qkv / o_proj / gate_up / down / lm_head / router

AttentionOpFactory
  prefill / decode / mixed / MLA / sparse

MoEOpFactory
  router / fused_moe / shared_expert / dispatch/combine

NcclCommunicatorProfile
  allreduce / allgather / reduce_scatter / alltoall / p2p

RuntimeOpFactory
  metadata_build / kv_cache_update / sampler / output_processor / cpu_scheduler
```

Factory 只生成 `VirtualOp`, 不查 DB。

## 7. CostRouter 与 Backends

### 7.1 判断数据源的位置

采用 measured data 还是 roofline, 统一在 `CostRouter` 判断。

不放在:

- OpFactory
- OperatorDB
- RooflineAnalyzer

推荐优先级:

```text
1. ModuleProfileBackend
2. OperatorDBBackend exact
3. OperatorDBBackend nearest/interpolated
4. CommunicationFormulaBackend
5. RooflineBackend
6. ConservativeFallback
```

首版可以简化为:

```text
OperatorDB exact hit
Communication formula for collective
Roofline fallback for compute
```

### 7.2 CostPolicy

策略集中到 `CostPolicy`:

```python
@dataclass
class CostPolicy:
    mode: str = "operator_db_first"
    allow_nearest: bool = False
    allow_cross_framework_version: bool = False
    allow_cross_execution_mode: bool = False
    enable_module_profile: bool = True
    enable_operator_db: bool = True
    enable_roofline_fallback: bool = True
```

典型模式:

```text
roofline_only
operator_db_first
require_operator_db
compare_measured_vs_roofline
module_profile_first
```

### 7.3 RooflineBackend

`RooflineBackend` 复用现有 `RooflineAnalyzer`, 但不让 `RooflineAnalyzer` 直接读 OperatorDB。

它只消费 `VirtualOp.formula`:

```text
flops
load_weight
load_act
store_act
load_kv_cache
store_kv_cache
precision
```

### 7.4 CommunicationFormulaBackend

通信公式从 compute roofline 中独立出来。

首版可包装现有 NCCL 公式:

```text
allreduce_time
alltoall_time
allgather_time
reducescatter_time
p2p_time
```

后续再由 measured collective DB 替代。

### 7.5 Eager / CUDAGraph Timeline Backend

首版 TTFT 可以先使用简单模式:

```text
VirtualOp -> device_time -> step sum
```

在完成初版 TTFT 验证后, 再加入 segment-level timeline backend。

核心思想:

```text
VirtualOp-level:
  OperatorDB / Roofline / Formula 只估 device_time

Segment-level:
  根据 execution mode 估 wall time
```

不要在每个 op 上直接线性累加 launch overhead。eager 下 CUDA kernel launch 是
CPU enqueue 与 GPU execution 的流水, 大 kernel 可以掩盖很多 launch 时间。更合理的是:

```text
eager segment:
  用 CPU enqueue / GPU execute pipeline 估 exposed launch overhead

cudagraph segment:
  sum(device_time_with_padded_shape) + graph_replay_overhead
```

伪代码:

```python
def eager_timeline(kernels):
    cpu_t = 0.0
    gpu_t = 0.0
    for k in kernels:
        cpu_t += k.launch_overhead_s
        start = max(gpu_t, cpu_t)
        gpu_t = start + k.device_time_s
        if k.sync_after:
            cpu_t = max(cpu_t, gpu_t) + k.sync_overhead_s
    return max(cpu_t, gpu_t)
```

graph segment:

```python
def graph_timeline(kernels, replay_overhead_s):
    return replay_overhead_s + sum(k.device_time_s for k in kernels)
```

这一层应放在 CostRouter / SegmentBackend, 不放在 `RooflineBackend`。`RooflineBackend`
仍然只负责计算 device time。这样可以在 TTFT 初版之后单独打开该模型, 对比:

```text
simple sum mode
eager pipeline mode
cudagraph replay mode
```

看它对 TTFT/TPOT 的修正效果。

## 8. OperatorDB

### 8.1 OperatorRecord

OperatorDB 存标准化 record:

```python
@dataclass(frozen=True)
class OperatorRecord:
    signature: OperatorSignature
    hardware: str
    framework: str
    framework_version: str
    execution_mode: str
    kernel_source: str
    latency_us_p50: float
    latency_us_p10: float
    latency_us_p90: float
    n_iters: int
    n_warmups: int
    confidence: float
    roofline_us: float | None
    roofline_gap: float | None
    source: dict[str, Any]
```

### 8.2 数据来源

```text
collector RawRecord JSONL
  -> collector_v2 importer
  -> OperatorRecord
  -> OperatorStore
```

首版 store:

```text
MemoryOperatorStore
JsonlOperatorStore
```

后续:

```text
SQLiteOperatorStore
```

### 8.3 查询规则

主键包含:

```text
hardware
framework
framework_version
execution_mode
kernel_source
op_kind
op_subtype
dtype
shape_signature
parallel_signature
runtime_signature
```

不包含:

```text
model
source_profile
case_id
timestamp
worker_id
```

模型只是 shape 来源, 不是 DB 分区。

## 9. Collector / Profiler

### 9.1 分层采集

```text
Layer 1: Operator microbench
  GEMM / attention / MoE / norm / embedding / collective

Layer 2: Module microbench
  dense block / attention module / MoE block / runtime op

Layer 3: Single-step framework profile
  vLLM synthetic step, 对齐 StepOpPlan

Layer 4: End-to-end serving benchmark
  真实 vLLM benchmark, 校准 TTFT/TPOT/throughput
```

### 9.2 Collector 目录

collector 保持独立:

```text
collector/
  cases/
    gemm.py
    attention.py
    moe.py
    collective.py
  profiles/
    qwen3_4b.py
    qwen3_30b_a3b.py
  runners/
    vllm/
    sglang/
  data/
    operator_db/
```

profile 是 shape source:

```text
--profiles qwen3_4b qwen3_30b_a3b
```

不是 DB key。

### 9.3 没有真实硬件时

支持三类数据:

```text
proxy scaling
roofline envelope
synthetic DB
```

输出必须标注:

```text
hardware_status = pre_silicon
source_mix = roofline/proxy/synthetic
confidence = low
```

## 10. vLLM Adapter 与 Virtual Execution

### 10.1 保留 vLLM-native 路线

继续保留:

```text
VirtualPlatform
VirtualWorker
VirtualModelRunner
StepExtractor
FakeOutputBuilder
```

这是本系统区别于 AIConfigurator 和 LLMServingSim 的核心。

### 10.2 VirtualModelRunner 流程

```text
execute_model(scheduler_output):
  workload = StepExtractor.extract(scheduler_output)
  step_shape = StepShape.from_workload(workload, execution_context)
  plan = ModelGraphTemplate.build_step(step_shape, factories)
  trace = CostRouter.estimate(plan)
  VirtualTimeExecutor.apply(trace)
  output = FakeOutputBuilder.build(workload, trace)
  MetricsRecorder.record(workload, plan, trace, output)
  return output
```

### 10.3 Fake Output

建议三种模式:

| 模式 | 用途 |
|---|---|
| fixed | 最稳定, 适合吞吐测试 |
| deterministic_hash | 评估 prefix/stop 行为 |
| replay | 与真实 trace 对齐 |

### 10.4 Virtual Time

支持:

```text
sleep mode
virtual clock mode
```

第一版先做 sleep mode, 后续加入 virtual clock 支撑大规模搜索。

## 11. 通信与拓扑

通信不混进 compute op。

统一建成 `collective` VirtualOp:

```text
allreduce
allgather
reduce_scatter
alltoall
p2p
```

由 `NcclCommunicatorProfile` 生成 descriptor, 由 `CommunicationFormulaBackend` 或
`OperatorDBBackend` 给出 latency。

关键字段:

```text
world_size
tp
ep
node_count
gpus_per_node
topology
algo
protocol
execution_mode
```

## 12. Metrics 与报告

报告必须覆盖四层:

### 12.1 请求级

- TTFT
- TPOT
- E2E latency
- output length
- queueing time
- prefill/decode time

### 12.2 Step 级

- step latency
- scheduled tokens
- prefill/decode/mixed
- KV cache pressure
- graph/eager mode
- bottleneck

### 12.3 Op 级

- op name
- op kind/subtype
- shape
- latency
- source
- match type
- roofline gap
- confidence

### 12.4 数据源覆盖率

- measured DB 覆盖比例
- module profile 覆盖比例
- roofline fallback 比例
- low-confidence op 列表

## 13. 配置搜索与实验管理

搜索能力作为后期外层系统, 不阻塞第一轮 TTFT baseline 和 OperatorDB 主链路。
当前阶段只要求单次仿真具备 search-ready 边界: `DeployConfig` 贯穿
ModelGraphTemplate、OpFactory、CostRouter、OperatorDB key 和报告 metadata。

部署空间搜索不应该把 vLLM scheduler 放在内循环里, 否则每个候选配置都接近
真实执行速度, 搜索成本过高。

后期可以提供三种模式:

- `core_search`: core 层轻量 scheduler 生成 `GlobalStepWorkload`, `VirtualClock`
  按 step cost 推进 simulator time, 不 sleep, 用于大规模配置搜索。
- `trace_replay`: 重放 vLLM/sglang 导出的固定 schedule trace, 用于 cost what-if
  和框架调度形状对齐。
- `framework_live_validation`: 接入真实 vLLM/sglang, 只验证 core search 选出的
  top-K 配置。

详细设计见 `docs/CORE_SEARCH_MODE_DESIGN.md`。

近期不实现:

- SearchSpace 枚举
- Pareto
- core search scheduler
- trace replay
- top-K live validation 自动闭环

候选参数:

- TP/DP/EP
- max_num_batched_tokens
- max_num_seqs
- chunked prefill
- cudagraph capture size
- KV cache block size
- attention backend
- MoE routing profile

输出:

- TTFT/TPOT/E2E/throughput
- SLA pass/fail
- Pareto frontier
- bottleneck breakdown
- source/confidence summary

## 14. 工程目录结构

目标结构:

```text
LLMInferSim/
  docs/
    LLMINFERSIM_SYSTEM_SOLUTION_V3.md
    OPERATOR_DB_REFACTOR_DRAFT.md
    OPERATOR_SCHEMA.md
    COLLECTOR_DESIGN.md

  collector/
    cli.py
    schemas.py
    scheduler.py
    harness.py
    writer.py
    checkpoint.py
    registry.py
    cases/
      gemm.py
      attention.py
      moe.py
      collective.py
    profiles/
      qwen3_4b.py
      qwen3_30b_a3b.py
    runners/
      vllm/
      sglang/
    data/

  llm_infer_sim/
    adapters/
      vllm/
      sglang/

    core/
      graph/
        virtual_op.py
        step_shape.py
        step_plan.py
        model_template.py
      models/
        qwen.py
        deepseek.py
        llama.py
      ops/
        factories/
          dense.py
          attention.py
          moe.py
          communication.py
          runtime.py
      routing/
        moe.py
      operator_schema/
        signature.py
        canonical.py
        gemm.py
        attention.py
        moe.py
        collective.py
      operator_db/
        schema.py
        query.py
        store.py
        loader.py
        importers/
          collector_v2.py
        stores/
          memory.py
          jsonl.py
          sqlite.py
      cost/
        router.py
        trace.py
        backends/
          module_profile.py
          operator_db.py
          formula.py
          roofline.py
          communication.py
      scheduler_sim/
        request.py
        config.py
        queue.py
        kv_capacity.py
        policy.py
        scheduler.py
        chunked_prefill.py
        admission.py
      replay/
        schedule_trace.py
        trace_reader.py
        trace_runner.py
      profiles/
      workload/
      planning/
      simulation/
        virtual_clock.py
      metrics/

    search/
      search_space.py
      runner.py
      pareto.py
      picking.py
      report.py
```

现有 `core/cost_model/*` 文件在迁移过程中仅作为公式参考; 新架构落地后可删除或归档到外部备份,
不作为运行时兼容路径保留。

## 15. 分阶段路线

### 阶段 1: 理想架构最小闭环

目标:

```text
Qwen3-4B dense
prefill/decode
TP=1
source=roofline
```

产物:

- `VirtualOp`
- `StepShape`
- `StepOpPlan`
- `QwenModelGraphTemplate`
- `DenseOpFactory`
- `AttentionOpFactory minimal`
- `RooflineBackend`
- `CostRouter`
- `StepCostTrace`

验收:

- 能输出 per-op trace。
- 每个 GEMM op 带 OperatorDB query 字段。
- 总时间与旧 `ModelCoreCostModel` 接近。

### 阶段 2: OperatorDB GEMM

目标:

```text
collector vllm_gemm record
  -> importer
  -> OperatorDB
  -> runtime VirtualOp exact hit
```

验收:

- GEMM exact hit。
- miss fallback roofline。
- eager/cudagraph 不互相命中。
- trace 显示 source/match/roofline_gap/case_id。

### 阶段 3: Qwen3-30B MoE

目标:

- `MoERoutingProfile`
- `MoEOpFactory`
- `fused_moe` OperatorDB lookup
- balanced / power_law routing

验收:

- balanced 与 power_law 是不同 signature。
- TP/EP 进入 key。
- decode/prefill 能生成 fused_moe op。

### 阶段 4: Collective 与 NCCL

目标:

- `NcclCommunicatorProfile`
- collective VirtualOp
- communication formula backend
- collective OperatorDB

验收:

- allreduce/alltoall 与 collector case 对齐。
- execution_mode 区分 framework overhead。
- topology/algo/protocol 进入 runtime。

### 阶段 5: Attention / ModuleProfile

目标:

- attention op-level profile
- mixed attention module profile
- ModuleProfileBackend
- derived profile / step cost cache

### 阶段 6: 配置搜索

目标:

- SearchRunner
- Pareto frontier
- SLA picking
- source/confidence-aware report

### 阶段 7: 真实芯片校准

目标:

- 迁移 collector 到自研芯片 runtime。
- 采 OperatorDB。
- 建 ModuleProfile。
- 与真实 serving benchmark 做误差闭环。

验收:

- measured DB 覆盖主要耗时 op。
- 常见 workload TTFT/TPOT 误差进入目标范围。

## 16. 风险与应对

### 16.1 vLLM 内部接口变化

应对:

- adapter 层隔离 vLLM 对象。
- core 不 import vLLM。
- step extractor 单独测试。

### 16.2 Fake output 改变调度行为

应对:

- fixed mode 默认避开 stop token。
- deterministic_hash/replay 用于特殊评估。
- 报告输出 fake policy。

### 16.3 OperatorDB 覆盖不足

应对:

- CostRouter fallback 到 roofline。
- 报告 source coverage。
- `require_operator_db` 模式用于采集覆盖率测试。

### 16.4 单算子组合误差

应对:

- 引入 ModuleProfileBackend。
- 支持 fused VirtualOp。
- 用 E2E benchmark 校准。

### 16.5 通信建模过粗

应对:

- communication formula 首版可用。
- 后续接 measured collective DB。
- topology signature 显式化。

### 16.6 cudagraph 建模过粗

应对:

- `execution_mode` 进入 key。
- runtime 记录 graph_capture_size / padded_tokens / fallback_reason。
- 后续细分 full/piecewise/fallback。

## 17. 最终形态

最终 LLMInferSim 应形成闭环:

```text
真实框架 scheduler
  -> framework-independent workload
  -> model op graph
  -> measured/profile/formula cost trace
  -> virtual execution
  -> request-level metrics
  -> collector/calibration feedback
  -> configuration search
```

它的核心能力不是单个 roofline 公式, 而是:

```text
真实调度 + 结构化 op graph + 数据驱动 cost backend + 可解释 trace
```
