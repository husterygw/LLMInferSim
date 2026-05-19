# LLMInferSim 重构详细设计

> 文档版本: 2026-05-18  
> 适用范围: LLMInferSim 当前 vLLM-native virtual backend 路线  
> 目标: 在保留真实 vLLM scheduler 的前提下, 将现有 roofline-first cost model 重构为 OperatorDB + ModuleProfile + Roofline fallback 的多层性能系统。

## 1. 背景与目标

当前 LLMInferSim 已经走通了最关键的方向:

```text
真实 vLLM scheduler
  -> VirtualPlatform / VirtualWorker / VirtualModelRunner
  -> VllmStepExtractor
  -> cost model
  -> VirtualTime / Metrics / FakeOutput
```

这个方向与系统设计一致: 不在 vLLM 外部重写 scheduler, 而是在 vLLM 原生执行链路中替换 GPU 执行耗时。

现有主要短板在 cost 层:

- `ModelCoreCostModel` 同时承担 phase dispatch、op 构造、roofline 分析、mixed attention 估计和 breakdown 聚合, 责任过重。
- roofline 是主路径, 实测数据只通过 `EfficiencyProfile` 局部修正, 缺少 AIC-style operator database。
- calibration 已经能产出 `dense.csv`、`per_sequence.csv`、`attention.csv`, 但 runtime 还没有把这些 module profile 作为一级 backend 使用。
- eager / cudagraph 差异目前主要简化成 kernel overhead 开关, 没有显式建模 graph dispatch、padding、graph replay 和 graph 外 runtime。
- workload IR 还偏聚合字段, 对 4D attention profile、operator lookup、communication event 的表达不够稳定。

本次重构目标:

```text
vLLM SchedulerOutput
  -> framework-neutral StepWorkload / BatchShape IR
  -> StepPlan / VirtualOp / CommEvent
  -> CostBackendRouter
       -> ModuleProfileBackend
       -> MeasuredOperatorDBBackend
       -> DerivedProfileBackend
       -> MeasuredCommBackend
       -> RooflineFallbackBackend
  -> GlobalStepCost
  -> VirtualClock / Metrics / FakeOutput
```

核心原则:

- vLLM adapter 只处理 vLLM 对象和版本差异。
- core 不 import vLLM。
- OperatorDB 用于底层真实性能和 roofline gap 分析。
- ModuleProfile 用于贴近 vLLM step/module 的高精度 runtime 估计。
- Roofline 始终保留为 miss fallback 和新硬件预估基础。
- cudagraph 是 execution mode, 不是简单的 "kernel overhead = 0"。

## 2. 非目标

本次重构不做:

- 不重写 vLLM scheduler。
- 不替代 vLLM KV block manager。
- 不要求第一版就接 ASTRA-Sim。
- 不要求第一版覆盖所有 speculative decoding、multimodal encoder graph、DBO microbatch 细节。
- 不删除旧路径, 第一阶段用 re-export 和 adapter 保持 examples/tests 可运行。

## 3. 目标目录结构

推荐最终结构:

```text
llm_infer_sim/
├── adapters/
│   └── vllm/
│       ├── virtual_platform.py
│       ├── virtual_worker.py
│       ├── virtual_model_runner.py
│       ├── step_extractor.py
│       ├── profile_extractor.py
│       ├── cudagraph_extractor.py
│       └── output_adapter.py
│
├── core/
│   ├── ir/
│   │   ├── workload.py
│   │   ├── batch_shape.py
│   │   ├── virtual_op.py
│   │   ├── execution_mode.py
│   │   └── step_plan.py
│   │
│   ├── planning/
│   │   ├── step_plan_builder.py
│   │   ├── op_builder.py
│   │   ├── attention_shape_builder.py
│   │   └── comm_event_builder.py
│   │
│   ├── cost/
│   │   ├── estimator.py
│   │   ├── result.py
│   │   ├── router.py
│   │   └── backends/
│   │       ├── base.py
│   │       ├── module_profile.py
│   │       ├── operator_db.py
│   │       ├── derived_profile.py
│   │       ├── comm_profile.py
│   │       └── roofline.py
│   │
│   ├── operator_db/
│   │   ├── schema.py
│   │   ├── database.py
│   │   ├── lookup.py
│   │   ├── roofline_compare.py
│   │   └── importers/
│   │       ├── csv.py
│   │       ├── aic.py
│   │       └── vllm_profile.py
│   │
│   ├── profiles/
│   │   ├── bundle.py
│   │   ├── model_config.py
│   │   ├── deploy_config.py
│   │   ├── hardware.py
│   │   ├── backend_profile.py
│   │   ├── module_profile.py
│   │   ├── operator_db_profile.py
│   │   ├── comm_db.py
│   │   ├── shape_buckets.py
│   │   └── efficiency_profile.py
│   │
│   ├── ops/
│   │   ├── base.py
│   │   ├── linear.py
│   │   ├── attention.py
│   │   ├── ffn.py
│   │   ├── moe.py
│   │   ├── normalization.py
│   │   ├── embedding.py
│   │   ├── communication.py
│   │   └── kv_transfer.py
│   │
│   ├── runtime/
│   │   ├── virtual_clock.py
│   │   ├── output_generator.py
│   │   ├── kv_block_tracker.py
│   │   └── pd_transfer.py
│   │
│   └── metrics/
│       ├── collector.py
│       ├── request_metrics.py
│       ├── step_metrics.py
│       └── reporter.py
│
├── calibration/
│   ├── runner.py
│   ├── engine.py
│   ├── batch.py
│   ├── shots.py
│   ├── catalog.py
│   ├── timings.py
│   ├── csv_io.py
│   ├── fit.py
│   └── models/
│
└── tools/
    ├── profile_convert.py
    ├── inspect_operator_db.py
    ├── compare_roofline.py
    └── compare_trace.py
```

兼容迁移建议:

```text
core/workload/workload.py          -> core/ir/workload.py
core/cost_model/cost_result.py     -> core/cost/result.py
core/cost_model/roofline.py        -> core/cost/backends/roofline.py
core/cost_model/model_core.py      -> core/cost/estimator.py
core/planning/plan_builder.py      -> core/planning/step_plan_builder.py
core/simulation/*                  -> core/runtime/*
core/profiles/profile_manager.py   -> core/profiles/bundle.py
```

旧模块先保留薄 wrapper:

```python
# llm_infer_sim/core/cost_model/model_core.py
from llm_infer_sim.core.cost.estimator import StepCostEstimator as ModelCoreCostModel
```

## 4. 分层职责

### 4.1 adapters/vllm

职责:

- 识别当前 vLLM version 的 `SchedulerOutput` 字段。
- 抽取 `ProfileBundle`。
- 抽取 `ExecutionContext`, 包括 cudagraph mode、capture sizes、max capture size、enforce eager。
- 将 vLLM runner output 适配为 fake output。
- 不做 cost 公式。

关键模块:

```text
step_extractor.py
  SchedulerOutput -> GlobalStepWorkload

profile_extractor.py
  VllmConfig -> ProfileBundle

cudagraph_extractor.py
  VllmConfig / SchedulerOutput summary -> ExecutionModeHints

virtual_model_runner.py
  orchestrate only:
    update request states
    extract workload
    estimate cost
    apply PD / DP sync
    record metrics
    sleep / fake output
```

### 4.2 core/ir

core IR 是框架无关层, 不 import vLLM。

#### GlobalStepWorkload

表达真实 scheduler step:

```python
@dataclass
class RequestWorkload:
    request_id: str
    phase: StepPhase
    num_tokens: int
    context_len: int
    target_output_len: int
    generated_tokens: int
    is_chunked: bool
    chunk_size: int
    prompt_len: int | None = None
    num_computed_before: int | None = None
    num_computed_after: int | None = None
```

```python
@dataclass
class GlobalStepWorkload:
    step_id: int
    phase: StepPhase
    requests: list[RequestWorkload]
    total_scheduled_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    num_prefix_cached_tokens: int
    execution: StepExecutionContext
```

#### BatchShape

新增稳定 shape:

```python
@dataclass(frozen=True)
class AttentionBatchShape:
    prefill_chunk: int
    kv_prefill: int
    n_decode: int
    kv_decode: int
    actual_tokens: int
    actual_reqs: int
    padded_tokens: int | None = None
    padded_reqs: int | None = None
    uniform_decode: bool = False
    execution_mode: ExecutionMode = ExecutionMode.EAGER
```

```python
@dataclass(frozen=True)
class DenseBatchShape:
    layer: str
    tokens: int
    padded_tokens: int | None
    phase: StepPhase
    execution_mode: ExecutionMode
```

```python
@dataclass(frozen=True)
class PerSequenceShape:
    layer: str
    sequences: int
    padded_sequences: int | None
    execution_mode: ExecutionMode
```

#### ExecutionMode

```python
class ExecutionMode(str, Enum):
    EAGER = "eager"
    CUDA_GRAPH_FULL = "cuda_graph_full"
    CUDA_GRAPH_PIECEWISE = "cuda_graph_piecewise"
    CUDA_GRAPH_NONE_FALLBACK = "cuda_graph_none_fallback"
```

注意:

- `CUDA_GRAPH_NONE_FALLBACK` 表示用户配置允许 cudagraph, 但当前 step 没命中 capture key。
- runtime reporting 需要区分 `EAGER` 和 fallback, 否则很难解释性能。

#### VirtualOp

OperatorDB 查询的统一输入:

```python
@dataclass(frozen=True)
class VirtualOp:
    op_id: str
    op_kind: OpKind
    op_name: str
    phase: StepPhase
    layer_idx: int | None
    shape: dict[str, int | float | str]
    dtype: DTypeSpec
    parallel: ParallelSpec
    backend: str
    kernel_source: str | None
    execution_mode: ExecutionMode
    fallback_profile: OperatorProfile
```

#### CommEvent

```python
@dataclass(frozen=True)
class CommEvent:
    collective: str
    bytes: int
    world_size: int
    group: str
    phase: StepPhase
    layer_idx: int | None
    execution_mode: ExecutionMode
    topology_hint: str
```

### 4.3 core/planning

`StepPlanBuilder` 负责把 workload + profile bundle 变成 `StepPlan`:

```text
GlobalStepWorkload
  -> BatchShape
  -> per-rank VirtualOp list
  -> CommEvent list
  -> RuntimeOp list
```

第一版可以保持 symmetric rank:

```python
@dataclass
class StepPlan:
    step_id: int
    phase: StepPhase
    workload: GlobalStepWorkload
    attention_shape: AttentionBatchShape
    dense_shapes: list[DenseBatchShape]
    per_sequence_shapes: list[PerSequenceShape]
    per_rank_ops: dict[int, list[VirtualOp]]
    comm_events: list[CommEvent]
    runtime_ops: list[RuntimeOp]
```

阶段演进:

- P0: `StepPlanBuilder` 包装现有 `build_mixed_plan`, pure prefill/decode 仍可走旧路径。
- P1: 所有 phase 都走 `StepPlanBuilder`。
- P2: per-rank ops 真实化, 支持 EP asymmetric 和 rank imbalance。

## 5. Cost Backend 体系

### 5.1 Backend 接口

```python
@dataclass
class CostQuery:
    op: VirtualOp | CommEvent | BatchShape
    bundle: ProfileBundle
    strict: bool = False

@dataclass
class CostLookupResult:
    latency_s: float
    source: str
    confidence: float
    matched_key: str | None
    roofline_s: float | None = None
    gap_to_roofline: float | None = None
    breakdown: dict = field(default_factory=dict)

class CostBackend(Protocol):
    def supports(self, query: CostQuery) -> bool: ...
    def estimate(self, query: CostQuery) -> CostLookupResult | None: ...
```

### 5.2 CostBackendRouter

查询优先级:

```text
1. ModuleProfileBackend
   - dense.csv
   - per_sequence.csv
   - attention.csv 4D

2. DerivedProfileBackend
   - 常见 step/module shape cache
   - 可由 OperatorDB 聚合生成

3. MeasuredOperatorDBBackend
   - AIC-style operator/kernel records
   - exact / interpolation / bucket lookup

4. MeasuredCommBackend
   - NCCL/custom allreduce/alltoall/p2p measured DB

5. RooflineFallbackBackend
   - 永远可用
```

router 返回结果必须带 source:

```text
module_profile:attention:exact
operator_db:gemm:interpolated
operator_db:attention:bucket
comm_db:allreduce:exact
roofline:fallback
```

### 5.3 StepCostEstimator

主入口:

```python
class StepCostEstimator:
    def estimate(self, plan: StepPlan) -> GlobalStepCost:
        # 1. query module-level attention if available
        # 2. query dense/per_sequence/module profile if available
        # 3. otherwise query per-op OperatorDB
        # 4. miss fallback roofline
        # 5. aggregate per-rank
        # 6. add graph/runtime/comm overhead
```

聚合逻辑:

```text
per-rank model_time = aggregate(ops)
per-rank comm_time = aggregate(comm_events)
per-rank runtime_time = runtime overhead
global latency = max(per_rank total)
```

第一版仍可用 symmetric rank:

```text
global latency = rank0 total
per_rank_costs = [rank0 copied N times]
```

## 6. OperatorDB 设计

### 6.1 为什么需要 OperatorDB

ModuleProfile 回答:

```text
这个 vLLM module/layer 在某个 batch shape 下花多久?
```

OperatorDB 回答:

```text
这个 kernel/op 在某个 m/n/k、dtype、backend、execution mode 下真实跑多久?
```

OperatorDB 的价值:

- measured 命中时提升预测精度。
- miss 时提供相似 shape interpolation。
- 分析 roofline gap, 指导 efficiency profile。
- 新硬件 bring-up 时从 microbench 快速积累数据。
- 对 AIC / LLMServingSim / 自研 collector 数据做统一承载。

### 6.2 Schema

```python
@dataclass(frozen=True)
class OperatorKey:
    op_kind: str              # gemm / attention / moe / norm / activation / comm
    op_name: str              # qkv_proj / o_proj / flash_attn / allreduce
    phase: str                # prefill / decode / mixed
    backend: str              # vllm / trtllm / sglang / torch
    kernel_source: str        # cublas / cutlass / triton / flash_attn / flashinfer / nccl
    input_dtype: str
    weight_dtype: str | None
    output_dtype: str | None
    kv_dtype: str | None
    execution_mode: str       # eager / cuda_graph_full / cuda_graph_piecewise
    device: str
    tp: int
    ep: int
```

```python
@dataclass
class OperatorShape:
    m: int | None = None
    n: int | None = None
    k: int | None = None
    tokens: int | None = None
    batch: int | None = None
    seq_len: int | None = None
    ctx_len: int | None = None
    prefill_chunk: int | None = None
    kv_prefill: int | None = None
    n_decode: int | None = None
    kv_decode: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    experts: int | None = None
    topk: int | None = None
    bytes: int | None = None
    world_size: int | None = None
    padded_tokens: int | None = None
    padded_reqs: int | None = None
```

```python
@dataclass
class OperatorRecord:
    key: OperatorKey
    shape: OperatorShape
    latency_us_p50: float
    latency_us_p90: float
    latency_us_mean: float
    samples: int
    roofline_us: float | None
    gap_to_roofline: float | None
    metadata: dict
```

### 6.3 存储格式

第一版用 CSV/Parquet 均可。推荐目录:

```text
configs/operator_db/
└── <device>/
    └── <backend>/
        └── <backend_version>/
            ├── gemm.csv
            ├── attention.csv
            ├── moe.csv
            ├── norm.csv
            ├── activation.csv
            ├── communication.csv
            └── meta.yaml
```

CSV 最小列:

```text
op_kind,op_name,phase,backend,kernel_source,
input_dtype,weight_dtype,output_dtype,kv_dtype,
execution_mode,device,tp,ep,
m,n,k,tokens,batch,seq_len,ctx_len,
prefill_chunk,kv_prefill,n_decode,kv_decode,
num_heads,num_kv_heads,head_dim,experts,topk,
bytes,world_size,padded_tokens,padded_reqs,
latency_us_p50,latency_us_p90,latency_us_mean,samples,
roofline_us,gap_to_roofline,
framework,version,cuda,driver,notes
```

### 6.4 Lookup 策略

查询优先级:

```text
exact:
  full key + exact shape

shape interpolation:
  same op_kind/backend/kernel/dtype/device/execution_mode
  interpolate along shape axes

bucket:
  same op_kind/backend/kernel/dtype/device
  shape bucket average efficiency

op-only efficiency:
  same op_kind/dtype/device

roofline fallback:
  compute from OperatorProfile
```

LookupResult 需要记录:

```text
lookup_level = exact | interpolated | bucket | op_default | roofline
matched_records
confidence
```

### 6.5 Roofline Gap

每条 measured record 都应计算:

```text
gap_to_roofline = measured_latency_us_p50 / roofline_us
efficiency = roofline_us / measured_latency_us_p50
```

分析工具输出:

```text
by op_kind
by kernel_source
by dtype
by tokens / m-n-k bucket
by prefill_chunk / kv / n_decode
by execution_mode
```

示例报告:

```text
op_kind=gemm kernel=cublas dtype=bf16
  m<=16:    gap p50=3.8x, p90=5.2x
  m 17-128: gap p50=1.9x
  m>1024:   gap p50=1.15x

op_kind=attention kernel=flashinfer mode=cuda_graph_full
  decode kv<=2k: gap p50=2.4x
  decode kv>16k: gap p50=1.3x
```

## 7. ModuleProfile 设计

### 7.1 数据来源

复用当前 calibration 输出, 对齐 LLMServingSim:

```text
dense.csv
  layer,tokens,time_us

per_sequence.csv
  layer,sequences,time_us

attention.csv
  prefill_chunk,kv_prefill,n_decode,kv_decode,time_us
```

建议扩展 meta.yaml:

```yaml
device: RTX_4090
model: Qwen/Qwen3-32B
dtype: bf16
backend: vllm
vllm_version: 0.20.1
tp: 1
ep: 1
execution_mode: eager
attention_backend: flashinfer
capture:
  cudagraph_mode: NONE
  capture_sizes: []
columns:
  attention_axes:
    - prefill_chunk
    - kv_prefill
    - n_decode
    - kv_decode
```

CUDA graph profile 应单独记录:

```yaml
execution_mode: cuda_graph_full
capture:
  padded_tokens: true
  capture_sizes: [1, 2, 4, 8, 16, ...]
```

### 7.2 Runtime 查询

`ModuleProfileBackend` 支持:

```text
AttentionBatchShape -> attention.csv
DenseBatchShape     -> dense.csv
PerSequenceShape    -> per_sequence.csv
```

attention 优先查 4D exact:

```text
(prefill_chunk, kv_prefill, n_decode, kv_decode)
```

miss 后:

```text
same prefill_chunk/n_decode, interpolate kv
same mode bucket
fallback OperatorDB attention
fallback roofline attention
```

### 7.3 ModuleProfile 与 OperatorDB 的关系

推荐 runtime 优先:

```text
ModuleProfileBackend > OperatorDBBackend > RooflineFallback
```

原因:

- ModuleProfile 更贴近真实 vLLM module/layer, 隐式包含 kernel fusion、runtime layout、backend dispatch。
- OperatorDB 更适合解释和迁移, 也适合作为 ModuleProfile miss fallback。

同时保留 DerivedProfile:

```text
OperatorDB records
  -> aggregate common module shapes
  -> write derived profile cache
```

## 8. CUDA Graph 建模

### 8.1 vLLM CUDA Graph 事实

当前 vLLM 中 cudagraph 不是完整 serving step 一张图。

大致边界:

```text
execute_model()
  preprocess / update states / prepare inputs          graph 外
  attention metadata / padding decision                graph 外
  set_forward_context(...)
  model forward                                        graph 内, FULL 或 PIECEWISE
  postprocess / compute_logits / sample state          graph 外
sample_tokens()                                        graph 外为主
```

模式:

```text
NONE
  eager

FULL
  capture/replay model forward for padded BatchDescriptor

PIECEWISE
  capture/replay torch.compile/piecewise subgraphs

FULL_DECODE_ONLY
  uniform decode 用 FULL, mixed/prefill 不用 FULL

FULL_AND_PIECEWISE
  uniform decode 可 FULL, mixed 可 PIECEWISE
```

### 8.2 ExecutionContext

新增:

```python
@dataclass
class StepExecutionContext:
    configured_mode: str
    runtime_mode: ExecutionMode
    actual_tokens: int
    actual_reqs: int
    padded_tokens: int
    padded_reqs: int | None
    capture_size: int | None
    max_capture_size: int | None
    uniform_decode: bool
    graph_replay_count: int
    graph_key: str | None
```

第一版可以由 adapter 近似推导:

- 如果 `enforce_eager=True` 或 cudagraph disabled: `EAGER`
- 如果 `total_tokens > max_cudagraph_capture_size`: `CUDA_GRAPH_NONE_FALLBACK`
- 如果 pure/uniform decode 且 full decode key 命中: `CUDA_GRAPH_FULL`
- 如果 mixed key 命中 piecewise: `CUDA_GRAPH_PIECEWISE`
- 否则 fallback none

长期更准确的方式:

- 在 vLLM adapter 中镜像 `CudagraphDispatcher.dispatch` 所需输入。
- 或在真实 vLLM runner 中把 `cudagraph_mode` / `batch_desc` 暴露给 virtual runner。

### 8.3 Cost 变化

eager:

```text
model_forward = sum(op compute/mem + per-kernel overhead)
runtime = input prep + metadata + logits + sampling
```

FULL cudagraph:

```text
model_forward =
  compute/mem on padded shape
  + graph_replay_overhead
  + captured comm/runtime if actually inside forward

outside_forward =
  input prep
  attention metadata build
  buffer copy/sync
  logits
  sampling
```

PIECEWISE:

```text
model_forward =
  sum(piece compute/mem)
  + num_pieces * graph_replay_overhead
  + non-captured op overhead
```

第一版近似:

```text
FULL:
  per-op kernel_overhead = 0
  graph_replay_overhead = hw.graph_replay_overhead_us
  shape = padded_tokens

PIECEWISE:
  per-op kernel_overhead = 0 for captured categories
  graph_replay_overhead = num_layers or num_pieces estimate
  dynamic attention/runtime ops still eager if not captured

NONE_FALLBACK:
  same as eager, but report source as cudagraph_fallback
```

### 8.4 Profile 要求

不要混用 eager profile 和 cudagraph profile。

profile key 必须包含:

```text
execution_mode
actual_tokens
padded_tokens
uniform_decode
capture_size
```

否则小 batch decode 会明显误差:

- eager 小 op launch overhead 高。
- cudagraph launch overhead 少, 但 padding 可能变多。
- graph 外 logits/sampling 仍存在。

## 9. Communication 建模

现有 `ops/communication.py` 可以作为公式 fallback。

重构后新增:

```text
MeasuredCommBackend
  -> exact measured NCCL/custom AR/alltoall/p2p
  -> interpolation by bytes/world_size
  -> formula fallback
```

CommEvent key:

```text
collective
bytes
world_size
group
device
topology_hint
execution_mode
backend
```

注意 cudagraph:

- 如果 collective 在 captured forward 内, framework call overhead 可减少。
- NCCL kernel 本身时间不消失。
- custom allreduce / flashinfer fused allreduce 需要通过 `kernel_source` 区分。

## 10. Metrics 与 Trace

现有 `MetricsCollector` 的方向正确: 使用 simulator time, 不用 wall-clock。

重构后扩展 step record:

```text
step_id
phase
actual_tokens
padded_tokens
batch_size
padded_reqs
execution_mode
cost_source_summary
total_latency
model_forward_time
runtime_time
comm_time
graph_replay_time
roofline_fallback_count
operator_db_hit_count
module_profile_hit_count
critical_rank
rank_imbalance
```

新增 trace 输出:

```json
{
  "step_id": 12,
  "phase": "mixed",
  "execution_mode": "cuda_graph_piecewise",
  "actual_tokens": 192,
  "padded_tokens": 256,
  "attention_shape": {
    "prefill_chunk": 128,
    "kv_prefill": 2048,
    "n_decode": 64,
    "kv_decode": 4096
  },
  "cost_sources": {
    "module_profile": 3,
    "operator_db": 24,
    "roofline": 2
  },
  "latency_s": 0.0123
}
```

## 11. 配置与开关

环境变量建议:

```text
LLM_INFER_SIM_TIME_MODE=realtime|instant
LLM_INFER_SIM_COST_BACKENDS=module,operator_db,comm,roofline
LLM_INFER_SIM_OPERATOR_DB=/path/to/operator_db
LLM_INFER_SIM_MODULE_PROFILE=/path/to/profile_bundle
LLM_INFER_SIM_STRICT_PROFILE=0|1
LLM_INFER_SIM_DUMP_COST_SOURCES=0|1
LLM_INFER_SIM_DUMP_STEP_TRACE=/path/to/trace.jsonl
LLM_INFER_SIM_CUDAGRAPH_MODEL=auto|force_eager|force_full|force_piecewise
```

ProfileBundle 增加:

```python
@dataclass
class ProfileBundle:
    model: ModelConfig
    deploy: DeployConfig
    hw: HardwareConfig
    efficiency: EfficiencyProfile
    backend: BackendExecutionProfile
    module_profile: ModuleProfileBundle | None = None
    operator_db: OperatorDatabaseConfig | None = None
    comm_db: CommDatabaseConfig | None = None
```

## 12. 迁移计划

### P0: 目录与兼容层

目标:

- 新建 `core/ir`, `core/cost`, `core/operator_db`。
- 移动或复制 dataclass, 旧路径 re-export。
- 不改变 runtime 行为。

验收:

```text
pytest tests/core tests/adapters/vllm
examples/run_platform_selected.py
examples/run_opt125m.py
```

### P1: ModuleProfile runtime backend

目标:

- 实现 `ModuleProfileBundle` loader。
- `attention.csv` 4D lookup 接入 mixed attention path。
- `dense.csv` / `per_sequence.csv` 可查, 但未命中仍走旧 cost。

验收:

- 构造测试 profile CSV, exact hit 返回 CSV latency。
- miss fallback 到 roofline。
- step breakdown 标出 `source=module_profile`。

### P2: OperatorDB schema 与 roofline gap

目标:

- 实现 `OperatorRecord` / `MeasuredOperatorDB`。
- 实现 CSV importer。
- 每条 record 计算或保存 roofline gap。
- `compare_roofline.py` 输出 gap report。

验收:

- GEMM exact lookup。
- attention exact lookup。
- miss fallback roofline。
- gap report by op_kind。

### P3: StepPlanBuilder 统一化

目标:

- pure prefill/decode/mixed 全部生成 `StepPlan`。
- `StepCostEstimator` 通过 router 查询 backend。
- `ModelCoreCostModel` 变成兼容 wrapper。

验收:

- 旧 tests 通过。
- examples 输出 latency 与旧路径在无 profile 时基本一致。

### P4: CUDA graph execution mode

目标:

- 加 `ExecutionMode` 和 `StepExecutionContext`。
- vLLM adapter 推导 runtime mode 和 padded shape。
- Cost 按 padded shape 估计 forward。
- graph replay overhead 单独计入。

验收:

- eager 与 cudagraph mode trace 可区分。
- `total_tokens > max_capture_size` 标为 fallback。
- padded_tokens 出现在 step trace。

### P5: Measured communication backend

目标:

- 接 `measure_collectives.py` 输出。
- allreduce/alltoall/p2p exact/interpolation lookup。
- comm formula fallback。

验收:

- TP/EP tests 中 comm source 可见。
- eager/cudagraph framework overhead 区分。

### P6: per-rank / EP asymmetric

目标:

- StepPlan per-rank ops 真实化。
- MoE expert routing skew 影响 rank imbalance。
- global step latency = max(rank total)。

验收:

- EP uneven case 中 `rank_imbalance > 0`。
- critical_rank 正确。

## 13. 测试策略

新增测试:

```text
tests/core/ir/test_batch_shape.py
tests/core/cost/test_backend_router.py
tests/core/cost/test_module_profile_backend.py
tests/core/operator_db/test_schema.py
tests/core/operator_db/test_lookup.py
tests/core/operator_db/test_roofline_compare.py
tests/adapters/vllm/test_cudagraph_extractor.py
```

关键 case:

- pure prefill shape。
- pure decode uniform shape。
- chunked prefill continuation。
- mixed prefill + decode 4D attention。
- prefix cache hit。
- cudagraph full decode padded token。
- cudagraph fallback because over max capture size。
- OperatorDB exact hit。
- OperatorDB miss roofline fallback。
- ModuleProfile exact hit优先于 OperatorDB。

## 14. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| vLLM cudagraph dispatch 内部变化 | mode/padding 推导漂移 | adapter 独立封装, 测试锁关键字段 |
| profile 与 runtime backend 不匹配 | 预测误差 | meta.yaml 强校验 backend/version/dtype/execution_mode |
| OperatorDB schema 过宽 | 初期开发慢 | 第一版只填常用列, 其余 nullable |
| ModuleProfile 与 OperatorDB 双重计算 | latency 重复 | router 明确优先级, plan 标记 consumed scopes |
| cudagraph graph 外开销漏算 | 小 batch 偏乐观 | runtime_overhead 单独 backend, trace 中独立展示 |
| 旧 tests/examples 大面积修改 | 迁移成本高 | 旧路径 re-export, 分阶段替换 |

## 15. 最小可用版本定义

MVP 完成标准:

```text
1. 现有 examples 仍能跑通。
2. mixed attention 能优先查 attention.csv 4D profile。
3. GEMM/attention 至少一种 op 能查 OperatorDB exact record。
4. OperatorDB miss 自动 fallback roofline。
5. step trace 能看到 cost source 和 roofline fallback count。
6. eager/cudagraph 在 trace 中有不同 execution_mode。
```

MVP 后系统形态:

```text
真实 vLLM scheduler
  + ModuleProfile for high-fidelity step/module cost
  + OperatorDB for kernel-level measured data and roofline gap
  + Roofline for fallback and new hardware what-if
  + VirtualMetrics for request-level TTFT/TPOT/E2E
```

## 16. 总结

这次重构的关键不是换入口, 而是把 cost 层从一个 roofline-first 单体模型改成可解释、可校准、可迁移的多层性能系统。

最终目标:

```text
不在 vLLM 外部模拟 vLLM;
在 vLLM 内部虚拟执行 vLLM。

不只给一个 latency;
还告诉你 latency 来自哪个 backend、哪个 op、和 roofline 差多少。
```
