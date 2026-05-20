# LLMInferSim 新架构实施方案

> 状态: draft  
> 日期: 2026-05-19  
> 基于: `LLMINFERSIM_SYSTEM_SOLUTION_V3.md`  
> 目标: 将理想新架构拆成可执行阶段。每个阶段说明目标、改动范围、复用代码、具体步骤、验收标准和测试建议。

## 0. 实施原则

### 0.1 直接切到新架构

本工程仍处于实验阶段, 且会在实施前备份原工程, 因此不维护双运行路径。

实施策略改为:

```text
直接建设新架构主路径
旧代码只作为迁移时的公式参考
不提供旧 runtime fallback
不保留引擎切换开关
```

新主路径:

```text
core/graph
core/models
core/ops/factories
core/operator_schema
core/operator_db
core/cost
```

现有 `core/cost_model/model_core.py`、`core/cost_model/layer_builder.py`、
`core/cost_model/mixed_attention.py` 可以在迁移过程中被阅读、复制公式或逐步删除,
但不再作为运行时兼容路径。

### 0.2 每阶段必须可验收

每个阶段都应该有明确产物:

```text
代码模块
单元测试
示例输出
```

不要出现“改了一堆抽象但无法运行”的阶段。

### 0.3 先 Qwen, 后 DeepSeek

第一批模型聚焦:

```text
Qwen3-4B dense
Qwen3-30B-A3B MoE
```

DeepSeek V3/V4、MLA、V4 sparse attention 放到后续阶段。

### 0.4 Operator schema 先于 OperatorDB

先保证:

```text
runtime VirtualOp
collector Case.params
OperatorDB Query
```

三者字段能对齐。

再接真实 DB。

### 0.5 Search-ready, not search-first

最佳部署配置搜索放到后期实现。当前阶段不提前做 `SearchRunner`、Pareto、
trace replay 或 live validation 自动闭环。

但新架构必须从第一阶段开始满足一个约束:

```text
单次仿真必须能被 DeployConfig 完整参数化。
```

也就是说, 现在先把核心链路做成 search-ready:

```text
DeployConfig
  -> ModelGraphTemplate / OpFactory
  -> VirtualOp.parallel/runtime
  -> CostRouter / OperatorDB key
  -> StepCostTrace / Report metadata
```

后期真正做空间搜索时, 应该只是外层循环:

```python
for candidate in candidates:
    deploy_config = candidate.to_deploy_config()
    result = simulator.run(workload, model_config, deploy_config)
    collect(result)
```

如果后期为了搜索还需要反向修改 `ops`、`operator_db`、`cost` 的核心边界,
说明当前 search-ready 约束没有落实好。

当前阶段明确不做:

```text
Pareto
SearchSpace 枚举
CoreScheduler 搜索模式
AIC-style static envelope
top-K live validation 自动闭环
```

### 0.6 测试策略: roofline-first, staged

测试不等全部重构完成后再补, 而是随每个阶段一起落地。当前工程主线是
roofline 层, 因此测试目标不是一开始追求端到端精确拟合, 而是先保证:

```text
公式正确
图构建正确
roofline lower bound 正确
实测差距可解释
```

测试分四类:

| 类型 | 是否需要芯片 | 进入普通 CI | 目的 |
|---|---:|---:|---|
| 架构/公式单测 | 否 | 是 | 锁住 VirtualOp、StepShape、RooflineBackend、DeployConfig 边界 |
| 模型图/step 单测 | 否 | 是 | 验证 Qwen/后续 MoE 的 op list、shape、parallel/runtime metadata |
| 算子级实测对比 | 是 | 否, 标记 hardware/integration | 生成 roofline gap, 对齐 collector 与 OperatorSignature |
| 模型级实测对比 | 是 | 否, 标记 hardware/integration | 对比 StepCostTrace 与真实 benchmark/profiler, 做误差归因 |

#### 测试随阶段安排

```text
阶段 1:
  DeployConfig / VirtualOp / StepShape / RooflineBackend 单测
  Qwen3-4B prefill/decode StepCostEngine smoke

阶段 2:
  OperatorSignature canonicalize 单测
  runtime VirtualOp 与 collector case 生成同一 signature

阶段 3:
  OperatorDB exact hit / miss fallback 单测
  GEMM roofline vs real compare report, hardware/integration

阶段 4:
  Qwen3-30B MoE graph / routing / fused_moe roofline 单测
  fused_moe roofline vs real compare report, hardware/integration

阶段 5:
  collective signature / communication formula 单测
  NCCL/collective roofline/formula vs real compare report, hardware/integration

阶段 6/7:
  Qwen3-4B 模型级 StepCostTrace vs real step/profiler 对比
  单请求 TTFT/TPOT smoke compare report

阶段 10:
  正式真实芯片校准回归, 输出稳定误差指标和补采建议
```

#### 第一批必须随阶段 1 一起加的测试

```text
tests/core/graph/test_virtual_op.py
tests/core/graph/test_step_shape.py
tests/core/profiles/test_deploy_config.py
tests/core/cost/test_roofline_backend.py
tests/core/cost/test_step_cost_engine_qwen.py
tests/core/models/test_qwen_dense_template.py
```

首批 smoke shapes 控制在 3 个:

```text
i128_o128      baseline
i2048_o128     prefill scaling
i128_o2048     decode scaling
```

#### 算子级实测对比报告

算子级对比不作为普通 CI 硬门槛, 先生成报告:

```text
OperatorRooflineCompareReport
  op_signature
  op_kind / op_subtype
  shape / dtype / parallel / runtime
  roofline_us
  real_p50_us
  real_p90_us
  gap = real_p50_us / roofline_us
  bottleneck = compute / memory
  arithmetic_intensity
  pass_level
```

首版 sanity:

```text
roofline_us <= real_p50_us * 1.1
```

如果 roofline 明显比真实还慢, 优先检查公式、bytes、硬件峰值和 dtype。
如果 gap 很大, 不立刻修正掉, 先作为 kernel efficiency / launch / layout /
routing / data movement 的诊断信号。

#### 模型级实测对比报告

模型级对比分两层:

```text
Step/device-level:
  StepCostTrace roofline total
  vs real profiler 中对应 step 的 GPU kernel sum

Serving-metrics-level:
  sim TTFT / TPOT / E2E
  vs vLLM/sglang benchmark TTFT / TPOT / E2E
```

roofline-only 阶段的验收口径:

```text
不要求端到端精确。
要求趋势正确、拆解正确、lower bound 正确、误差可归因。
```

## 1. 阶段 1: 图表示最小闭环

### 1.1 目标

建立新架构最小闭环:

```text
GlobalStepWorkload
  -> StepShape
  -> QwenModelGraphTemplate
  -> StepOpPlan
  -> CostRouter
  -> RooflineBackend
  -> StepCostTrace
```

同时完成 search-ready 最小要求:

```text
同一个 workload, 不改核心代码, 只替换 DeployConfig,
就能生成不同 parallel/runtime metadata 的 StepCostTrace。
```

覆盖范围:

```text
Qwen3-4B dense
prefill / decode
TP=1
source=roofline
```

### 1.2 新增模块

```text
llm_infer_sim/core/graph/
  __init__.py
  virtual_op.py
  step_shape.py
  step_plan.py

llm_infer_sim/core/profiles/
  deploy.py

llm_infer_sim/core/cost/
  __init__.py
  trace.py
  router.py
  backends/
    __init__.py
    roofline.py

llm_infer_sim/core/models/
  __init__.py
  qwen.py

llm_infer_sim/core/ops/factories/
  __init__.py
  dense.py
  attention.py
  normalization.py
  embedding.py
```

### 1.3 复用代码

| 现有代码 | 复用方式 |
|---|---|
| `core/workload/workload.py` | `StepShape.from_workload()` 输入 |
| `core/profiles/profile_manager.py` | `ProfileBundle` 继续作为上下文 |
| `core/profiles/model_config.py` | `QwenModelGraphTemplate` 输入 |
| `core/profiles/deploy.py` | TP/dtype/batch 配置 |
| `core/profiles/hardware.py` | RooflineBackend 输入 |
| `core/cost_model/roofline.py` | `RooflineBackend` 内部调用 |
| `core/ops/linear.py` | Dense formula 来源 |
| `core/ops/attention.py` | Attention formula 来源 |
| `core/ops/normalization.py` | norm/activation formula 来源 |
| `core/ops/embedding.py` | embedding/lm_head formula 来源 |
| `core/cost_model/layer_builder.py` | Qwen dense op 顺序和公式迁移参考 |

### 1.4 具体步骤

#### Step 1.0 收敛 `DeployConfig`

文件:

```text
core/profiles/deploy.py
```

目标不是实现搜索, 而是把单次仿真的部署维度固定成一个对象。

首版字段:

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

要求:

- `StepShape` 可读取 `execution_mode`、`block_size` 等 runtime 字段。
- `OpFactory` 只从 `DeployConfig` 读取 TP/EP/DP/MoE 并行信息。
- `VirtualOp.parallel/runtime` 必须携带这些字段, 供 OperatorDB key 使用。
- `StepCostTrace` metadata 中保留 deploy config 摘要。

暂不做:

- candidate 枚举
- Pareto
- 多配置 runner

#### Step 1.1 定义 `VirtualOp`

文件:

```text
core/graph/virtual_op.py
```

实现:

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

`formula` 首版字段:

```text
flops
load_weight
load_act
store_act
load_kv_cache
store_kv_cache
op_precision
comm_bytes
comm_type
op_category
```

#### Step 1.2 定义 `StepShape`

文件:

```text
core/graph/step_shape.py
```

实现:

```python
StepShape.from_workload(workload, bundle) -> StepShape
```

首版支持:

- PREFILL
- DECODE

MIXED / CHUNKED_PREFILL 先返回 NotImplemented, 后续阶段在新架构中实现。

#### Step 1.3 定义 `StepOpPlan`

文件:

```text
core/graph/step_plan.py
```

实现:

```python
@dataclass(frozen=True)
class StepOpPlan:
    step_id: int
    phase: str
    ops: tuple[VirtualOp, ...]
    metadata: dict[str, Any]
```

#### Step 1.4 实现 `RooflineBackend`

文件:

```text
core/cost/backends/roofline.py
```

做法:

1. 将 `VirtualOp.formula` 转为现有 `OperatorProfile`。
2. 调用 `RooflineAnalyzer.analyze()`。
3. 返回 `CostTraceEntry`。

注意:

- 不要让 `RooflineAnalyzer` 知道 `VirtualOp`。
- 转换逻辑集中在 backend 内部。

#### Step 1.5 实现 `CostTraceEntry / StepCostTrace`

文件:

```text
core/cost/trace.py
```

字段:

```text
op_name
op_kind
op_subtype
latency_s
source
match_type
roofline_s
roofline_gap
metadata
```

可提供:

```python
StepCostTrace.to_report_dict()
```

用于当前 reporter/tests 消费, 但不承诺兼容旧 dict 结构。

#### Step 1.6 实现 `CostRouter`

文件:

```text
core/cost/router.py
```

首版逻辑:

```text
for op in plan.ops:
  if op_kind == collective:
    skip or formula backend later
  else:
    RooflineBackend.estimate(op)
aggregate -> StepCostTrace
```

`source` 统一为:

```text
roofline
```

#### Step 1.7 实现 Dense/Norm/Embedding Factories

文件:

```text
core/ops/factories/dense.py
core/ops/factories/normalization.py
core/ops/factories/embedding.py
```

先实现 Qwen dense path 需要的 op:

- `attn_norm`
- `qkv_proj`
- `rope`
- `o_proj`
- `attn_add`
- `mlp_norm`
- `gate_up_proj`
- `mlp_act`
- `down_proj`
- `mlp_add`
- `embedding`
- `lm_head`

公式从现有:

```text
core/ops/linear.py
core/ops/normalization.py
core/ops/embedding.py
core/ops/attention.py
```

迁移或包装。

#### Step 1.8 实现 Minimal Attention Factory

文件:

```text
core/ops/factories/attention.py
```

首版支持:

- Qwen GQA prefill attention
- Qwen GQA decode attention

可以先生成一个 `op_kind=attention` 的 fused op, formula 复用现有
`attention_prefill_flash/attention_decode_flash` 聚合结果。

#### Step 1.9 实现 `QwenModelGraphTemplate`

文件:

```text
core/models/qwen.py
```

方法:

```python
build_step(step_shape, bundle, factories) -> StepOpPlan
```

首版每层生成:

```text
attn_norm
qkv_proj
rope
attention
o_proj
attn_add
mlp_norm
gate_up_proj
mlp_act
down_proj
mlp_add
```

暂不支持:

- TP allreduce
- MoE
- mixed
- MLA

#### Step 1.10 新增新入口 `StepCostEngine`

文件:

```text
core/cost/engine.py
```

或:

```text
core/engine/step_cost_engine.py
```

接口:

```python
class StepCostEngine:
    def __init__(self, bundle: ProfileBundle):
        ...

    def estimate(self, workload: GlobalStepWorkload) -> StepCostTrace:
        step = StepShape.from_workload(workload, self.bundle)
        plan = self.template.build_step(step, self.bundle, self.factories)
        return self.router.estimate(plan)
```

### 1.5 测试

新增:

```text
tests/core/graph/test_virtual_op.py
tests/core/graph/test_step_shape.py
tests/core/cost/test_roofline_backend.py
tests/core/cost/test_step_cost_engine_qwen.py
```

测试点:

1. `StepShape.from_workload()` 正确处理 prefill/decode。
2. Qwen3-4B prefill 生成 op list。
3. `qkv_proj/gate_up_proj/down_proj` shape 正确。
4. `RooflineBackend` 对 `VirtualOp` 输出非零 latency。
5. total_time 在量级和主要 op 占比上与现有公式预期一致。

### 1.6 验收标准

必须满足:

```text
Qwen3-4B prefill/decode TP=1 可跑通
StepCostTrace 有 per-op entries
所有 GEMM op 带 op_kind/op_subtype/shape/parallel/runtime
source 全部为 roofline
同一 workload 仅替换 DeployConfig 后, VirtualOp.parallel/runtime 和 trace metadata 随之变化
总时间量级和主要 op 占比可解释
```

## 2. 阶段 2: Operator Schema Contract

### 2.1 目标

建立 collector/runtime/OperatorDB 的共同字段契约。

此阶段不一定接真实 DB, 但必须保证:

```text
collector Case.params
runtime VirtualOp
OperatorDB Query

能 canonicalize 到同一个 OperatorSignature。
```

### 2.2 新增模块

```text
llm_infer_sim/core/operator_schema/
  __init__.py
  signature.py
  canonical.py
  gemm.py
  attention.py
  moe.py
  collective.py
```

### 2.3 具体步骤

#### Step 2.1 定义 `OperatorSignature`

文件:

```text
core/operator_schema/signature.py
```

字段:

```python
@dataclass(frozen=True)
class OperatorSignature:
    op_kind: str
    op_subtype: str
    dtype: str
    shape: tuple[tuple[str, Any], ...]
    parallel: tuple[tuple[str, Any], ...]
    runtime: tuple[tuple[str, Any], ...]
```

提供:

```python
stable_hash()
to_json_dict()
```

#### Step 2.2 GEMM canonicalizer

文件:

```text
core/operator_schema/gemm.py
```

实现:

```python
gemm_case_params_to_signature(params, runtime_context)
gemm_virtual_op_to_signature(op, context)
```

字段必须对齐 collector:

```text
op_subtype, m, n, k, dtype, tp
```

#### Step 2.3 MoE canonicalizer

文件:

```text
core/operator_schema/moe.py
```

字段:

```text
num_tokens
hidden
moe_intermediate
topk
num_experts
routing_distribution
power_law_alpha
dtype
tp
ep
```

#### Step 2.4 Collective canonicalizer

文件:

```text
core/operator_schema/collective.py
```

字段:

```text
op_subtype
message_bytes
dtype
world_size
tp
ep
node_count
gpus_per_node
backend
algo
protocol
topology
```

#### Step 2.5 Attention canonicalizer

文件:

```text
core/operator_schema/attention.py
```

首版字段:

```text
op_subtype
num_tokens
num_seqs
q_len
kv_len
num_q_heads
num_kv_heads
head_dim
dtype
kv_dtype
tp
attention_backend
```

#### Step 2.6 Runtime VirtualOp 接 schema

所有 factory 生成的 `VirtualOp` 必须能:

```python
virtual_op_to_signature(op, context)
```

### 2.4 测试

新增:

```text
tests/core/operator_schema/test_gemm_signature.py
tests/core/operator_schema/test_moe_signature.py
tests/core/operator_schema/test_collective_signature.py
```

测试点:

1. collector GEMM `Case.params` 与 runtime GEMM `VirtualOp` signature 相同。
2. MoE balanced 与 power_law signature 不同。
3. eager 与 cudagraph signature 不同。
4. framework_version / kernel_source 进入 runtime key。

### 2.5 验收标准

```text
Qwen3-4B runtime GEMM ops 可生成 OperatorSignature
collector gemm case 可生成 OperatorSignature
同 shape 同 runtime context signature 相同
不同 execution_mode signature 不同
```

## 3. 阶段 3: OperatorDB GEMM exact hit

### 3.1 目标

将 collector 采集的 GEMM JSONL 数据接入 runtime。

路径:

```text
collector RawRecord
  -> collector_v2 importer
  -> OperatorRecord
  -> OperatorStore
  -> OperatorDBBackend
  -> CostRouter
```

### 3.2 新增模块

```text
llm_infer_sim/core/operator_db/
  __init__.py
  schema.py
  query.py
  store.py
  loader.py
  importers/
    __init__.py
    collector_v2.py
  stores/
    __init__.py
    memory.py
    jsonl.py

llm_infer_sim/core/cost/backends/
  operator_db.py
```

### 3.3 具体步骤

#### Step 3.1 定义 OperatorRecord

文件:

```text
core/operator_db/schema.py
```

字段:

```text
signature
hardware
framework
framework_version
execution_mode
kernel_source
latency_us_p50/p10/p90
n_iters/n_warmups
confidence
source
```

#### Step 3.2 定义 Store 接口

文件:

```text
core/operator_db/store.py
```

接口:

```python
class OperatorStore:
    def add(record: OperatorRecord) -> None: ...
    def lookup(signature: OperatorSignature) -> OperatorRecord | None: ...
```

#### Step 3.3 实现 MemoryOperatorStore

文件:

```text
core/operator_db/stores/memory.py
```

用于单测和小规模调试。

#### Step 3.4 实现 collector_v2 importer

文件:

```text
core/operator_db/importers/collector_v2.py
```

支持:

- GEMM RawRecord

将:

```text
RawRecord.params
RawRecord.framework
RawRecord.framework_version
RawRecord.execution_mode
RawRecord.kernel_source
RawRecord.device
RawRecord.metrics
RawRecord.metadata
```

转成 `OperatorRecord`。

#### Step 3.5 实现 JsonlOperatorStore/Loader

文件:

```text
core/operator_db/stores/jsonl.py
core/operator_db/loader.py
```

支持从:

```text
collector/data/operator_db/<hardware>/<framework-version>/gemm.jsonl
```

加载。

#### Step 3.6 实现 OperatorDBBackend

文件:

```text
core/cost/backends/operator_db.py
```

逻辑:

```text
VirtualOp -> OperatorSignature -> store.lookup
hit -> CostTraceEntry(source=operator_db, match_type=exact)
miss -> None
```

#### Step 3.7 CostRouter 接入 OperatorDBBackend

更新:

```text
core/cost/router.py
```

策略:

```text
operator_db_first:
  try operator_db
  miss -> roofline

roofline_only:
  skip operator_db

require_operator_db:
  miss -> error
```

### 3.4 测试

新增:

```text
tests/core/operator_db/test_collector_v2_importer.py
tests/core/operator_db/test_memory_store.py
tests/core/cost/test_operator_db_backend.py
tests/core/cost/test_cost_router_policy.py
```

测试点:

1. 手造 GEMM RawRecord 可导入 OperatorRecord。
2. Runtime GEMM VirtualOp exact hit。
3. miss fallback roofline。
4. `require_operator_db` miss 报错。
5. eager/cudagraph 不互相命中。

### 3.5 验收标准

```text
collector vllm_gemm 输出可被 runtime 加载
Qwen3-4B 至少 qkv_proj 或 gate_up_proj exact hit
trace 显示 source=operator_db, match_type=exact
miss op 仍 source=roofline
```

## 4. 阶段 4: Qwen3-30B MoE

### 4.1 目标

支持 Qwen3-30B-A3B 的 MoE op graph 与 fused_moe OperatorDB 查询。

覆盖:

```text
router
fused_moe
shared_expert optional
EP dispatch/combine descriptor
balanced / power_law routing
```

### 4.2 新增/修改模块

```text
core/routing/
  moe.py

core/ops/factories/
  moe.py

core/models/
  qwen.py

core/operator_schema/
  moe.py

core/cost/backends/
  operator_db.py
  roofline.py
```

### 4.3 具体步骤

#### Step 4.1 定义 MoERoutingShape

文件:

```text
core/routing/moe.py
```

字段:

```text
distribution
power_law_alpha
tokens_per_expert
tokens_per_rank
max_tokens_per_expert
max_tokens_per_rank
mean_tokens_per_expert
mean_tokens_per_rank
```

#### Step 4.2 实现 MoERoutingProfile

支持:

```text
balanced
power_law(alpha=1.01)
power_law(alpha=1.2)
```

首版可以不用真实 logits, 先生成 deterministic synthetic distribution。

复用:

```text
core/cost_model/moe_routing.py::estimate_distinct_experts
```

作为 fallback。

#### Step 4.3 实现 MoEOpFactory

生成:

- `router` GEMM VirtualOp
- `fused_moe` VirtualOp
- `shared_expert_*` GEMM VirtualOp
- `ep_alltoall_dispatch` collective VirtualOp
- `ep_alltoall_combine` collective VirtualOp

`fused_moe` shape 必须与 collector `cases/moe.py` 对齐。

#### Step 4.4 更新 QwenModelGraphTemplate

逻辑:

```text
if model.is_moe_layer(layer_idx):
  build_moe_layer
else:
  build_dense_layer
```

#### Step 4.5 MoE Roofline fallback

`fused_moe` 没有 DB hit 时, 使用 formula fallback:

```text
flops = tokens * topk * 3 * 2 * hidden * moe_intermediate / ep
weight bytes = distinct_experts * 3 * hidden * moe_intermediate * w_byte / ep
activation bytes = tokens_per_rank * hidden * a_byte
```

公式参考:

```text
layer_builder._build_moe_ffn_block
```

#### Step 4.6 MoE OperatorDB exact hit

collector MoE RawRecord 导入后, `fused_moe` VirtualOp 能 exact hit。

### 4.4 测试

新增:

```text
tests/core/routing/test_moe_routing_profile.py
tests/core/ops/test_moe_factory.py
tests/core/models/test_qwen_moe_template.py
tests/core/operator_schema/test_moe_signature.py
```

测试点:

1. balanced / power_law signature 不同。
2. Qwen3-30B-A3B MoE layer 生成 fused_moe。
3. EP>1 时生成 dispatch/combine collective op。
4. fused_moe miss fallback roofline。

### 4.5 验收标准

```text
Qwen3-30B-A3B prefill/decode 可生成 MoE StepOpPlan
fused_moe 带完整 schema fields
balanced/power_law 不互相命中
source 可为 operator_db 或 roofline
```

## 5. 阶段 5: Collective 与 NCCL Profile

### 5.1 目标

将通信从 compute formula 中独立出来, 形成 collective VirtualOp + backend。

### 5.2 新增/修改模块

```text
core/ops/factories/communication.py
core/cost/backends/communication.py
core/operator_schema/collective.py
```

### 5.3 具体步骤

#### Step 5.1 定义 Group/Topology 描述

可先简单定义:

```python
@dataclass(frozen=True)
class CommGroup:
    world_size: int
    tp: int
    ep: int
    dp: int
    node_count: int
    gpus_per_node: int
```

#### Step 5.2 实现 NcclCommunicatorProfile

文件:

```text
core/ops/factories/communication.py
```

输出 collective VirtualOp:

- allreduce
- alltoall
- allgather
- reduce_scatter
- p2p

#### Step 5.3 实现 CommunicationFormulaBackend

文件:

```text
core/cost/backends/communication.py
```

复用:

```text
core/ops/communication.py
```

现有公式:

- `allreduce_time`
- `alltoall_time`
- `allgather_time`
- `reducescatter_time`
- `p2p_time`

#### Step 5.4 CostRouter 接入 communication backend

逻辑:

```text
if op_kind == collective:
  try OperatorDBBackend
  miss -> CommunicationFormulaBackend
```

#### Step 5.5 collector collective 对齐

确保 collector collective case fields 与 runtime collective VirtualOp signature 对齐。

### 5.4 测试

新增:

```text
tests/core/ops/test_communication_factory.py
tests/core/cost/test_communication_backend.py
tests/core/operator_schema/test_collective_signature.py
```

测试点:

1. allreduce VirtualOp 生成正确。
2. `execution_mode=eager` 有 framework overhead。
3. `execution_mode=cudagraph` framework overhead 为 0。
4. collective signature 包含 topology/runtime 字段。

### 5.5 验收标准

```text
TP>1 时 Qwen dense layer 生成 allreduce op
EP>1 时 MoE layer 生成 alltoall op
collective op 可由 formula backend 估算
trace source=comm_formula 或 operator_db
```

## 6. 阶段 6: Attention / Mixed / ModuleProfile

### 6.1 目标

把 attention 从临时公式迁到标准 VirtualOp / ModuleProfile 路线。

覆盖:

- prefill attention
- decode attention
- mixed split
- mixed unified/ragged
- module profile backend

### 6.2 具体步骤

#### Step 6.1 AttentionOpFactory 完整化

支持:

```text
standard MHA/GQA
MLA placeholder
sparse placeholder
prefill
decode
mixed_split
mixed_unified
```

#### Step 6.2 MixedAttentionEstimator 迁移

将当前:

```text
core/cost_model/mixed_attention.py
```

逻辑迁移为:

```text
AttentionOpFactory builds mixed attention VirtualOp(s)
CostRouter estimates them
```

#### Step 6.3 ModuleProfile 数据模型

新增:

```text
core/cost/backends/module_profile.py
core/profiles/module_profile.py
```

支持:

```text
attention module profile
moe module profile
dense block profile
runtime profile
```

#### Step 6.4 CostRouter module priority

策略:

```text
module_profile_first:
  if module profile covers op/group -> use it
  else operator_db
  else roofline
```

首版可以先只支持 attention module profile。

### 6.3 测试

新增:

```text
tests/core/ops/test_attention_factory.py
tests/core/cost/test_module_profile_backend.py
tests/core/cost/test_mixed_attention_v2.py
```

### 6.4 验收标准

```text
mixed workload 可通过 StepCostEngine 估算 attention
trace 可区分 split/unified
attention source 可为 module_profile/operator_db/roofline
```

## 6.5 阶段 6.5: Eager/CUDAGraph Timeline 增强

### 6.5.1 目标

该阶段不作为初版 TTFT 的前置条件。建议在完成第一版 TTFT 测试后再加入, 用来评估
eager launch pipeline 与 cudagraph replay 对 TTFT/TPOT 的修正效果。

要解决的问题:

```text
eager 下不能简单逐 kernel 累加 launch overhead。
很多 CPU launch 时间会被前一个 kernel 的 GPU execution 掩盖。
只有 tiny kernels 或 CPU enqueue 跟不上 GPU 时, launch overhead 才暴露为 wall time。
```

### 6.5.2 建模分层

分三层:

```text
VirtualOp-level:
  OperatorDB / Roofline / Formula 输出 device_time

ExecutionSegment-level:
  eager pipeline 或 cudagraph replay 计算 segment wall time

Step-level:
  汇总 segments 得到 step latency
```

### 6.5.3 新增数据结构

新增:

```text
core/graph/execution_segment.py
core/cost/backends/timeline.py
```

建议结构:

```python
@dataclass(frozen=True)
class ExecutionSegment:
    segment_id: str
    mode: str                    # eager / cuda_graph_replay
    ops: tuple[VirtualOp, ...]
    replay_overhead_s: float = 0.0
```

```python
@dataclass(frozen=True)
class KernelTiming:
    op_name: str
    device_time_s: float
    launch_overhead_s: float
    sync_after: bool = False
    sync_overhead_s: float = 0.0
```

### 6.5.4 Eager pipeline 模型

不要用:

```text
sum(device_time) + kernel_count * launch_overhead
```

改用 CPU enqueue / GPU execute timeline:

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

输出 trace 里建议记录:

```text
device_time_s
launch_work_s
exposed_launch_overhead_s
segment_wall_time_s
```

### 6.5.5 CUDAGraph replay 模型

graph segment:

```python
def cuda_graph_timeline(kernels, replay_overhead_s):
    return replay_overhead_s + sum(k.device_time_s for k in kernels)
```

注意:

- cudagraph 可能使用 padded shape。
- StepShape/VirtualOp 需要记录 logical shape 和 padded shape。
- 不在 graph 内的 runtime ops 仍然走 eager segment。

### 6.5.6 接入策略

新增 CostPolicy:

```text
timeline_mode = simple_sum | eager_pipeline | cuda_graph_segments
```

初版默认:

```text
simple_sum
```

完成 TTFT baseline 后再开启:

```text
eager_pipeline
cuda_graph_segments
```

对比三组结果, 观察 TTFT/TPOT 修正幅度。

### 6.5.7 测试

新增:

```text
tests/core/cost/test_timeline_backend.py
```

测试点:

1. 大 kernel 能掩盖后续 launch overhead。
2. 连续 tiny kernels 暴露明显 launch overhead。
3. cudagraph segment 只加一次 replay overhead。
4. graph 外 eager runtime ops 仍走 eager pipeline。

### 6.5.8 验收标准

```text
timeline backend 可独立打开/关闭
simple_sum 与 eager_pipeline 结果可对比
cudagraph replay segment 支持 padded shape
trace 能报告 exposed_launch_overhead
```

## 7. 阶段 7: vLLM Adapter 接新引擎

### 7.1 目标

让 vLLM virtual execution 使用新 `StepCostEngine`。

### 7.2 具体步骤

#### Step 7.1 VirtualModelRunner 调用新引擎

替换前:

```text
VirtualModelRunner -> ModelCoreCostModel
```

改为:

```text
VirtualModelRunner -> StepCostEngine
```

直接替换, 不保留双模式。

#### Step 7.2 MetricsRecorder 适配 StepCostTrace

将 `StepCostTrace` 转为当前 metrics/reporter 需要的字段。

#### Step 7.3 Fake output 不变

输出生成逻辑保持不变, 只替换 latency source。

### 7.3 测试

新增/更新:

```text
tests/adapters/vllm/test_virtual_model_runner_step_cost_engine.py
tests/core/test_step_extractor.py
```

### 7.4 验收标准

```text
vLLM virtual runner 使用 StepCostEngine 跑 toy workload
metrics 输出 step/op/source trace
```

## 8. 阶段 8: Collector 扩展与覆盖率闭环

### 8.1 目标

让 collector 覆盖 runtime 生成的关键 `VirtualOp`。

### 8.2 具体步骤

#### Step 8.1 Coverage report

新增工具:

```text
scripts/report_operator_db_coverage.py
```

输入:

```text
StepOpPlan trace
OperatorDB store
```

输出:

```text
exact hit %
miss ops
miss by op_kind
miss by shape
recommended collector cases
```

#### Step 8.2 GEMM coverage

确保 Qwen3-4B dense GEMM shapes 都能生成 collector cases。

#### Step 8.3 MoE coverage

确保 Qwen3-30B-A3B MoE shapes 能生成 collector cases。

#### Step 8.4 Collective coverage

补 collective collector runner。

#### Step 8.5 Attention coverage

补 attention smoke cases, 复杂 mixed/module profile 后置。

### 8.3 验收标准

```text
Qwen3-4B dense GEMM 关键耗时 op 有 DB hit
Qwen3-30B MoE fused_moe 有 DB hit 或明确 miss reason
coverage report 能指导下一轮采集
```

## 9. 阶段 9: 配置搜索与实验管理

### 9.1 目标

在新 cost engine 稳定之后, 再增加 core-native 配置搜索。搜索内循环不依赖
vLLM scheduler, 不通过 `sleep` 推进时间, 而是由 LLMInferSim 自己的轻量
scheduler 生成 `GlobalStepWorkload`, 再用 `VirtualClock` 按 step cost 推进
simulator time。

本阶段是后期功能, 不属于第一轮 TTFT baseline 的必要路径。前面阶段只要求
`DeployConfig` 贯穿单次仿真链路, 让这里的搜索实现成为外层循环, 不反向修改
`ops/operator_db/cost` 核心边界。

详细方案见 `docs/CORE_SEARCH_MODE_DESIGN.md`。

### 9.2 新增模块

```text
llm_infer_sim/core/scheduler_sim/
  __init__.py
  request.py
  config.py
  queue.py
  kv_capacity.py
  policy.py
  scheduler.py
  chunked_prefill.py
  admission.py

llm_infer_sim/core/simulation/
  virtual_clock.py

llm_infer_sim/core/replay/
  schedule_trace.py
  trace_reader.py
  trace_runner.py

llm_infer_sim/search/
  __init__.py
  search_space.py
  runner.py
  pareto.py
  picking.py
  report.py
```

### 9.3 具体步骤

#### Step 9.1 Core search MVP

实现:

- `SearchRequest`
- `SimRequestState`
- `SchedulerConfig`
- `VirtualClock`
- `CoreScheduler`
- `CoreSearchRunner`

先不做 KV 容量、chunked prefill、prefix cache。cost backend 可以先用 mock cost,
用于验证请求生命周期和指标计算。

#### Step 9.2 请求生命周期与指标

跑通:

- arrival admission
- prefill -> decode
- decode 每 step 产出 1 token
- request finish
- TTFT / TPOT / E2E / throughput

此阶段应复用 `core/workload/request_state.py` 中 simulator-time 语义。

#### Step 9.3 接入真实 StepCostEngine

把 `CoreScheduler` 输出的 `GlobalStepWorkload` 接入:

```text
ModelOpBuilder -> VirtualOp -> CostRouter -> StepCostResult
```

要求:

- 同一份 workload schema 可用于 core search 和 trace replay。
- result 保留 source/confidence/breakdown。
- `StepCostResult.total_latency_s` 驱动 `VirtualClock.advance()`。

#### Step 9.4 KVCapacityState

实现 core-native KV 容量约束:

- block_size / num_gpu_blocks
- prefill 前判断能否 reserve
- decode 跨 block 时扩容
- finished 后 release
- peak KV utilization

首版只做 admission control, 不做 preemption/eviction。

#### Step 9.5 Chunked prefill 与 mixed batching

实现:

- `enable_chunked_prefill`
- `prefill_chunk_size`
- `max_num_partial_prefills`
- decode-first mixed policy

重点观察 TTFT 与 TPOT 的 tradeoff。

#### Step 9.6 Trace replay 与 top-K validation

实现:

- vLLM/sglang trace -> `GlobalStepWorkload`
- 固定 trace replay
- core search top-K 配置交给 framework live validation

core search 用于大规模搜索, live validation 用于最终校准。

### 9.4 搜索参数

首批:

```text
tp
dp
ep
max_num_batched_tokens
max_num_seqs
chunked_prefill
cudagraph capture size
kv_cache block size
attention backend
moe routing profile
```

### 9.5 输出

```text
TTFT
TPOT
E2E latency
throughput
SLA pass/fail
source coverage
confidence
bottleneck
Pareto frontier
```

### 9.6 验收标准

```text
同一模型多个 TP/EP 配置可在 core_search mode 下快速比较
搜索内循环不依赖 vLLM scheduler
搜索内循环不 sleep
输出 Pareto frontier
报告包含 source/confidence
top-K 配置可进入 vLLM/sglang live validation
```

## 10. 阶段 10: 真实芯片校准

### 10.1 目标

当目标硬件可用后, 建立真实数据闭环。

### 10.2 步骤

1. 环境锁定:
   - driver
   - runtime
   - framework version
   - clock/power mode
2. 跑 collector smoke:
   - GEMM
   - MoE
   - collective
   - attention
3. 导入 OperatorDB。
4. 跑 Qwen3-4B/Qwen3-30B 仿真。
5. 跑真实 vLLM benchmark。
6. 对比:
   - TTFT
   - TPOT
   - throughput
   - op coverage
   - roofline gap
7. 针对误差补采:
   - module profile
   - runtime profile
   - attention mixed profile

### 10.3 验收标准

```text
OperatorDB 覆盖主要耗时 op
source coverage report 可解释主要 miss
常见 workload 误差进入预设目标
```

## 11. 建议实施顺序

推荐实际执行顺序:

```text
1. 阶段 1: 图表示最小闭环
2. 阶段 2: Operator Schema Contract
3. 阶段 3: OperatorDB GEMM exact hit
4. 阶段 4: Qwen3-30B MoE
5. 阶段 5: Collective 与 NCCL Profile
6. 阶段 7: vLLM Adapter 接新引擎
7. 阶段 8: Collector 覆盖率闭环
8. 阶段 6: Attention / ModuleProfile
9. 阶段 6.5: Eager/CUDAGraph Timeline 增强
10. 阶段 9: 配置搜索
11. 阶段 10: 真实芯片校准
```

注意:

- 如果想尽快跑 vLLM 端到端, 先做阶段 7。
- 如果想先完善模型精度, 先做阶段 6。
- 阶段 6.5 建议放在初版 TTFT baseline 之后, 用于比较 timeline 模型带来的修正。

## 12. 第一轮建议任务包

第一轮建议只做到一个非常小但完整的闭环:

```text
Qwen3-4B prefill/decode
VirtualOp graph
RooflineBackend
StepCostTrace
```

任务清单:

1. 新建 `core/graph`。
2. 新建 `core/cost/trace.py`。
3. 新建 `core/cost/backends/roofline.py`。
4. 新建 `core/cost/router.py`。
5. 新建 `core/ops/factories/dense.py`。
6. 新建 `core/ops/factories/attention.py` minimal。
7. 新建 `core/models/qwen.py`。
8. 新建 `StepCostEngine`。
9. 同步写第一批测试:
   - `tests/core/graph/test_virtual_op.py`
   - `tests/core/graph/test_step_shape.py`
   - `tests/core/profiles/test_deploy_config.py`
   - `tests/core/cost/test_roofline_backend.py`
   - `tests/core/cost/test_step_cost_engine_qwen.py`
   - `tests/core/models/test_qwen_dense_template.py`
10. 用 `i128_o128`、`i2048_o128`、`i128_o2048` 做 Qwen3-4B
    prefill/decode smoke 和量级 sanity check。

这一步完成后, 再进入 Operator Schema 和 OperatorDB。
