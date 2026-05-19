# LLMInferSim 理想新架构方案: Op Graph + OperatorDB

> 状态: v3 draft  
> 日期: 2026-05-19  
> 方向: 以理想新架构为目标, 将 LLMInferSim 重构为 `scheduler step -> op graph -> multi-backend cost trace`。

## 1. 一句话目标

LLMInferSim 不再是一个以 layer 公式为中心的 cost model, 而是:

```text
从真实/拟真的 scheduler step 生成可查询的 op graph,
再由多级 cost backend 给出 latency、source、roofline gap 和 trace。
```

核心链路:

```text
Framework Adapter
  -> GlobalStepWorkload
  -> StepShape
  -> ModelGraphTemplate
  -> StepOpPlan
  -> CostRouter
  -> StepCostTrace
  -> Framework Output
```

这条链路里:

- 模型结构只负责生成 op。
- op descriptor 必须能对齐 collector case 和 OperatorDB query。
- CostBackend 负责查实测数据或走公式 fallback。
- Roofline 是 fallback 和 gap baseline, 不是全系统中心。

## 2. 设计原则

### 2.1 Op Graph 是中心

新的中心对象是 `VirtualOp` 和 `StepOpPlan`, 不是 `layer_builder.py` 和
`dense_layer_time/moe_layer_time`。

```text
旧:
  layer_builder builds ops and computes time

新:
  ModelGraphTemplate builds ops
  CostRouter computes time
```

### 2.2 Runtime Ops 必须能对齐 Collector

这是硬约束:

```text
Runtime VirtualOp descriptor
  必须 canonicalize 成
Collector Case.params / OperatorDB Query
```

也就是三者同构:

```text
collector Case.params
operator_db OperatorRecord.shape/parallel/runtime
runtime VirtualOp.shape/parallel/runtime
```

### 2.3 Factory 不查 DB

Factory 只生成 op descriptor。

```text
Factory:
  model/deploy/backend/context -> VirtualOp

CostBackend:
  VirtualOp -> measured latency or formula latency
```

不要让 `DenseOpFactory` / `MoEOpFactory` 内部查 OperatorDB。

### 2.4 Collector 与 Core 解耦

collector 继续作为独立数据生产线:

```text
collector -> raw JSONL -> importer -> OperatorDB
```

collector 不 import `llm_infer_sim.core`。

但两边必须共享同一份 operator schema contract。可以通过文档 + 测试 + canonicalizer 保证,
不要求 Python import 强耦合。

### 2.5 模型是 Shape 来源, 不是 DB 主键

Qwen3-4B / Qwen3-30B-A3B / DeepSeek 只用于生成 shape profiles 或 runtime ops。

OperatorDB 主键不包含 model name。

`source_model` / `source_profiles` 只进入 provenance。

### 2.6 eager / cudagraph 是硬边界

`execution_mode` 必须进入 OperatorDB query key。

默认不允许 eager query 命中 cudagraph record, 也不允许反向命中。

## 3. 目标目录结构

建议最终工程结构:

```text
LLMInferSim/
  README.md
  pyproject.toml

  docs/
    ARCHITECTURE.md
    OPERATOR_SCHEMA.md
    OPERATOR_DB_REFACTOR_DRAFT.md
    COLLECTOR_DESIGN.md

  collector/
    README.md
    __init__.py
    __main__.py
    cli.py
    registry.py
    scheduler.py
    harness.py
    writer.py
    checkpoint.py
    env_check.py
    version_resolver.py
    paths.py
    schemas.py

    cases/
      __init__.py
      gemm.py
      attention.py
      moe.py
      collective.py

    profiles/
      __init__.py
      registry.py
      qwen3_4b.py
      qwen3_30b_a3b.py

    runners/
      __init__.py
      vllm/
        __init__.py
        gemm.py
        attention.py
        moe.py
        collective.py
      sglang/
        __init__.py
        gemm.py
        attention.py
        moe.py
        collective.py

    importers/
      __init__.py
      validate_jsonl.py

    data/
      operator_db/
        <hardware>/
          <framework>-<version>/
            gemm.jsonl
            attention.jsonl
            moe.jsonl
            collective.jsonl
            errors/
            checkpoints/
            manifest.yaml

    tests/

  llm_infer_sim/
    __init__.py

    adapters/
      __init__.py
      base.py
      vllm/
        __init__.py
        profile_extractor.py
        step_extractor.py
        virtual_platform.py
        virtual_worker.py
        virtual_model_runner.py
      sglang/
        __init__.py
        profile_extractor.py
        step_extractor.py
        virtual_backend.py

    core/
      __init__.py
      graph/
      models/
      ops/
      routing/
      cost/
      operator_schema/
      operator_db/
      profiles/
      workload/
      planning/
      simulation/
      metrics/

  tests/
    core/
    adapters/
    operator_schema/
    operator_db/
    collector/
```

其中 `llm_infer_sim/core/` 建议最终结构:

```text
llm_infer_sim/core/
  graph/
    __init__.py
    virtual_op.py
    step_shape.py
    step_plan.py
    model_template.py

  models/
    __init__.py
    qwen.py
    deepseek.py
    llama.py

  ops/
    factories/
      __init__.py
      dense.py
      attention.py
      moe.py
      communication.py
      runtime.py

  routing/
    __init__.py
    moe.py

  cost/
    __init__.py
    router.py
    trace.py
    backends/
      __init__.py
      module_profile.py
      operator_db.py
      formula.py
      roofline.py
      communication.py

  operator_schema/
    __init__.py
    signature.py
    gemm.py
    attention.py
    moe.py
    collective.py

  operator_db/
    __init__.py
    schema.py
    query.py
    key.py
    store.py
    loader.py
    importers/
      collector_v2.py
    stores/
      memory.py
      jsonl.py
      sqlite.py

  profiles/
    __init__.py
    profile_bundle.py
    model_config.py
    deploy.py
    hardware.py
    backend_profile.py
    efficiency_profile.py
    sizing.py
    model_adapters/
      __init__.py
      qwen.py
      deepseek_v3.py
      deepseek_v4.py
      opt.py

  workload/
    __init__.py
    workload.py
    request_state.py

  planning/
    __init__.py
    planner.py
    stage_plan.py
    rank_plan.py
    execution_context.py

  simulation/
    __init__.py
    time_emulator.py
    output_generator.py
    kv_block_allocator.py

  metrics/
    __init__.py
    collector.py
    breakdown.py
    reporter.py
```

目录职责:

| 目录 | 职责 |
|---|---|
| `collector/` | 独立数据采集系统, 生产 raw JSONL |
| `collector/cases/` | 可测算子 shape 定义 |
| `collector/profiles/` | 从模型配置派生 shape profile |
| `collector/runners/` | vLLM/SGLang/TRT-LLM 真实 runner |
| `core/graph/` | `VirtualOp`, `StepShape`, `StepOpPlan` |
| `core/models/` | 模型结构模板, 只生成 op, 不算 latency |
| `core/ops/factories/` | 根据模型/step/backend 生成可查询 `VirtualOp` |
| `core/routing/` | MoE routing profile / routing shape |
| `core/operator_schema/` | collector/runtime/OperatorDB 的字段契约 |
| `core/operator_db/` | 实测算子数据存储、导入、查询 |
| `core/cost/` | CostRouter 和多 backend latency 估算 |
| `core/profiles/` | ProfileBundle、ModelConfig、DeployConfig、HardwareConfig |
| `core/workload/` | framework-independent scheduler workload |
| `core/planning/` | stage/rank/execution context 计划结构 |
| `core/simulation/` | 时间模拟、输出生成、KV block allocator |
| `core/metrics/` | breakdown/reporter/metrics aggregation |
| `core/cost_model/*` | 迁移期公式参考; 新架构落地后可删除或归档到外部备份 |

现有目录的定位:

```text
core/cost_model/layer_builder.py
  公式迁移参考

core/cost_model/roofline.py
  可复用为 RooflineBackend 内部实现

core/profiles/*
  保留, ProfileBundle 仍是配置入口

core/workload/*
  保留, GlobalStepWorkload 仍是 adapter 输出

collector/*
  保持独立采集系统
```

## 4. 核心对象

### 4.1 StepShape

`GlobalStepWorkload` 是 adapter/scheduler 视角。`StepShape` 是 graph/cost 视角。

```python
@dataclass(frozen=True)
class StepShape:
    step_id: int
    phase: str                    # prefill / decode / mixed / chunked_prefill

    total_tokens: int
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefill_requests: int = 0
    num_decode_requests: int = 0

    max_context_len: int = 0
    max_prefill_seqlen: int = 0
    avg_decode_context_len: int = 0

    request_shapes: tuple[dict, ...] = ()

    execution_mode: str = "eager"       # eager / cudagraph / cuda_graph_fallback
    graph_capture_size: int | None = None
    padded_tokens: int | None = None
```

首版可以从 `GlobalStepWorkload` 直接转换。

### 4.2 VirtualOp

`VirtualOp` 是新架构中心。它既能查 OperatorDB, 也能生成 roofline fallback 输入。

```python
@dataclass(frozen=True)
class VirtualOp:
    name: str
    op_kind: str                 # gemm / attention / moe / collective / norm / runtime
    op_subtype: str              # qkv_proj / fused_moe / allreduce / ...

    phase: str
    layer_idx: int | None

    dtype: str
    shape: dict[str, Any]
    parallel: dict[str, Any]
    runtime: dict[str, Any]

    formula: dict[str, Any]      # flops, memory bytes, comm bytes, precision
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
```

`shape/parallel/runtime/dtype/op_kind/op_subtype` 用于 OperatorDB query。  
`formula` 用于 roofline / communication formula fallback。

### 4.3 StepOpPlan

```python
@dataclass(frozen=True)
class StepOpPlan:
    step_id: int
    phase: str
    ops: tuple[VirtualOp, ...]
    metadata: dict[str, Any]
```

后续多卡可扩展成:

```text
StepOpPlan
  StagePlan[]
    RankPlan[]
      VirtualOp[]
```

首版先用 flat ops, 因为当前 `ModelCoreCostModel` 的输出也是 flat `per_op` breakdown。

## 5. Operator Schema Contract

这是 collector、runtime、OperatorDB 之间的唯一契约。

### 5.1 Contract 规则

所有可测算子必须定义:

```text
op_kind
op_subtype
dtype
shape fields
parallel fields
runtime fields
canonical signature
```

并提供 canonicalization:

```python
case_params_to_signature(op_kind, params) -> OperatorSignature
virtual_op_to_signature(op) -> OperatorSignature
record_to_signature(record) -> OperatorSignature
```

验收规则:

```text
collector 生成的 Case.params
runtime 生成的 VirtualOp
importer 生成的 OperatorRecord

对同一个物理算子必须得到同一个 OperatorSignature。
```

### 5.2 GEMM Contract

Collector case params:

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

Runtime VirtualOp:

```python
VirtualOp(
    op_kind="gemm",
    op_subtype="qkv_proj",
    dtype="bf16",
    shape={"m": 128, "n": 6144, "k": 2560},
    parallel={"tp": 1},
    runtime={
        "framework": "vllm",
        "framework_version": "0.19.1",
        "execution_mode": "eager",
        "kernel_source": "vllm_qkv_parallel_linear",
    },
    ...
)
```

Canonical key fields:

```text
op_kind=gemm
op_subtype
dtype
m,n,k
tp
framework
framework_version
execution_mode
kernel_source
hardware
```

首批 GEMM subtype:

```text
qkv_proj
o_proj
gate_up_proj
down_proj
lm_head
router
shared_expert_up_gate
shared_expert_down
```

### 5.3 MoE Contract

Collector case params:

```python
{
    "num_tokens": 128,
    "hidden": 2048,
    "moe_intermediate": 768,
    "topk": 8,
    "num_experts": 128,
    "tp": 1,
    "ep": 4,
    "routing_distribution": "power_law",
    "power_law_alpha": 1.2,
    "dtype": "bf16",
}
```

Runtime VirtualOp:

```python
VirtualOp(
    op_kind="moe",
    op_subtype="fused_moe",
    dtype="bf16",
    shape={
        "num_tokens": 128,
        "hidden": 2048,
        "moe_intermediate": 768,
        "topk": 8,
        "num_experts": 128,
        "routing_distribution": "power_law",
        "power_law_alpha": 1.2,
    },
    parallel={"tp": 1, "ep": 4},
    runtime={
        "framework": "vllm",
        "execution_mode": "eager",
        "kernel_source": "vllm_fused_moe",
    },
    ...
)
```

MoE key 必须包含 routing distribution。balanced 与 power_law 不能互相命中。

### 5.4 Attention Contract

首版建议 contract:

```python
{
    "op_subtype": "prefill" | "decode" | "mixed_unified" | "mixed_split",
    "num_tokens": ...,
    "num_seqs": ...,
    "q_len": ...,
    "kv_len": ...,
    "num_q_heads": ...,
    "num_kv_heads": ...,
    "head_dim": ...,
    "dtype": "bf16",
    "kv_dtype": "bf16",
    "tp": ...,
    "attention_backend": "flash_attn" | "flashinfer" | "paged_attention",
}
```

后续再加入:

- MLA latent dims
- sparse index topk
- block/page size
- ragged batch descriptor
- graph capture size

### 5.5 Collective Contract

```python
{
    "op_subtype": "allreduce" | "allgather" | "reduce_scatter" | "alltoall" | "p2p",
    "message_bytes": ...,
    "dtype": "bf16",
    "world_size": ...,
    "tp": ...,
    "ep": ...,
    "node_count": ...,
    "gpus_per_node": ...,
    "backend": "nccl",
    "algo": "auto" | "ring" | "tree" | ...,
    "protocol": "auto" | "LL" | "LL128" | "Simple",
    "topology": "...",
}
```

首版可以 `algo/protocol=auto`, 但字段要保留。

## 6. ModelGraphTemplate

模型模板只描述结构, 不算 latency, 不查数据库。

```python
class ModelGraphTemplate:
    def __init__(self, model: ModelConfig):
        self.model = model

    def build_step(self, step: StepShape, factories: FactoryBundle) -> StepOpPlan:
        ...
```

Qwen dense 层:

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

Qwen MoE 层:

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

DeepSeek 模板负责表达:

- MLA
- shared experts
- V3/V4 sparse attention
- hash routing
- hyper-connection

但这些模板仍然只生成 `VirtualOp`。

## 7. OpFactory 与 Profile

### 7.1 DenseOpFactory

职责:

- 生成 GEMM VirtualOp。
- 填充 GEMM contract 字段。
- 同时填充 roofline formula。

不负责:

- 查 OperatorDB。
- 做 latency fallback。

### 7.2 AttentionOpFactory

职责:

- 根据 model attention kind 和 step phase 生成 attention VirtualOp。
- 对 mixed step 生成 split 或 unified descriptor。
- 填充 attention contract。

当前 `MixedAttentionEstimator` 的逻辑可以迁移到这里:

```text
split_kernels -> 两个 attention ops
unified_ragged -> 一个 merged attention op
```

### 7.3 MoERoutingProfile

MoE routing 应从 scalar skew 升级为 routing shape。

```python
@dataclass(frozen=True)
class MoERoutingShape:
    distribution: str
    power_law_alpha: float
    tokens_per_expert: tuple[int, ...]
    tokens_per_rank: tuple[int, ...]
    max_tokens_per_expert: int
    max_tokens_per_rank: int
    mean_tokens_per_expert: float
    mean_tokens_per_rank: float
```

首版支持:

```text
balanced
power_law(alpha=1.01)
power_law(alpha=1.2)
```

这与 collector MoE cases 对齐。

### 7.4 MoEOpFactory

职责:

- 生成 router GEMM / hash lookup。
- 生成 fused_moe VirtualOp。
- 生成 shared expert GEMM。
- 生成 dispatch/combine collective VirtualOp。

`fused_moe` 的 shape 必须与 collector MoE case 对齐。

### 7.5 NcclCommunicatorProfile

通信 profile 负责描述可用算法、协议、拓扑、带宽和 latency model。

```python
class NcclCommunicatorProfile:
    def __init__(self, topology, env):
        self.available_algos = build_available_algos(topology, env)
        self.available_protocols = build_available_protocols(topology, env)
        self.latency_table = build_latency_model(topology)
        self.bandwidth_table = build_bandwidth_model(topology)
```

它输出 collective VirtualOp, 不直接返回最终 latency。最终 latency 由:

```text
OperatorDBBackend
  or CommunicationFormulaBackend
```

决定。

## 8. CostRouter 与 Backends

统一入口:

```python
class CostRouter:
    def estimate(self, plan: StepOpPlan) -> StepCostTrace:
        ...
```

优先级:

```text
1. ModuleProfileBackend
2. OperatorDBBackend exact
3. OperatorDBBackend nearest/interpolated
4. CommunicationFormulaBackend
5. RooflineBackend
```

首版可以简化为:

```text
OperatorDB exact hit
Communication formula for collective
Roofline fallback for compute
```

### 8.1 CostTraceEntry

```python
@dataclass(frozen=True)
class CostTraceEntry:
    op_name: str
    op_kind: str
    op_subtype: str
    latency_s: float
    source: str                  # module_profile / operator_db / comm_formula / roofline
    match_type: str              # exact / nearest / fallback
    roofline_s: float | None
    roofline_gap: float | None
    metadata: dict[str, Any]
```

### 8.2 StepCostTrace

```python
@dataclass(frozen=True)
class StepCostTrace:
    step_id: int
    phase: str
    total_latency_s: float
    compute_time_s: float
    memory_time_s: float
    comm_time_s: float
    entries: tuple[CostTraceEntry, ...]
    bottleneck: str
```

这比当前 dict 更适合长期演进。对外可提供 `to_report_dict()` 供 reporter 消费。

## 9. OperatorDB

### 9.1 OperatorRecord

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

### 9.2 Store

首版:

```text
MemoryOperatorStore
JsonlOperatorStore
```

后续:

```text
SQLiteOperatorStore
```

### 9.3 Importer

```text
collector RawRecord
  -> collector_v2 importer
  -> OperatorRecord
```

importer 的核心不是猜字段, 而是调用 operator schema canonicalizer。

## 10. Collector 关系

collector 继续按 op/case/profile 组织:

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
    vllm_gemm.py
    ...
```

关键约束:

```text
collector cases define measurable operator shapes
runtime VirtualOps instantiate those same measurable shapes
OperatorDB stores those same shapes after normalization
```

collector profile 是 shape source:

```text
--profiles qwen3_4b qwen3_30b_a3b
```

而不是 DB key:

```text
--models
```

## 11. 与现有 core 的关系

现有文件定位:

```text
core/cost_model/layer_builder.py
  公式迁移参考

core/cost_model/model_core.py
  旧入口, 新架构落地后由 StepCostEngine 替代

core/cost_model/roofline.py
  复用到 RooflineBackend

core/ops/*.py
  公式迁移来源, 后续逐步被 factories 替代

core/profiles/*
  保留, ProfileBundle 仍作为 builder/router 输入

core/workload/*
  保留, adapter 输出仍是 GlobalStepWorkload
```

不建议继续在 `layer_builder.py` 上增强复杂特性。新能力直接进入新架构模块。

## 12. 首版理想架构闭环

第一阶段目标:

```text
Qwen3-4B dense
prefill / decode
TP=1
source=roofline
```

实现:

```text
VirtualOp
StepShape
StepOpPlan
QwenModelGraphTemplate
DenseOpFactory
AttentionOpFactory minimal
RooflineBackend
CostRouter
StepCostTrace
```

验收:

- StepCostEngine 能输出 per-op trace。
- 每个 GEMM op 都带 OperatorDB query 所需字段。
- total time 在量级和主要 op 占比上与现有公式预期一致。
- trace 中每个 op 有 `source=roofline`。

## 13. 第二阶段: 接 OperatorDB GEMM

目标:

```text
collector vllm_gemm record
  -> importer
  -> OperatorDB
  -> runtime VirtualOp exact hit
```

验收:

- qkv/o/gate_up/down/lm_head 至少一种能 exact hit。
- miss 时 fallback 到 roofline。
- eager/cudagraph 不互相命中。
- trace 显示:

```text
source=operator_db
match_type=exact
roofline_gap=...
case_id=...
```

## 14. 第三阶段: Qwen3-30B MoE

目标:

```text
MoERoutingProfile
MoEOpFactory
fused_moe OperatorDB lookup
```

验收:

- balanced / power_law_1.01 / power_law_1.2 是不同 signature。
- TP/EP 进入 parallel key。
- Qwen3-30B-A3B decode/prefill 能生成 fused_moe VirtualOp。
- OperatorDB miss 时用 formula/roofline fallback。

## 15. 第四阶段: Collective 与 NCCL Profile

目标:

```text
NcclCommunicatorProfile
collective VirtualOp
CommunicationFormulaBackend
collective OperatorDB
```

验收:

- allreduce/alltoall op 与 collector collective case 对齐。
- `execution_mode=eager/cudagraph` 区分 framework overhead。
- topology/protocol/algo 字段进入 runtime。

## 16. 后续阶段

后续再做:

- DeepSeek MLA / V4 sparse attention template。
- attention module profile。
- mixed attention module profile。
- OperatorDB nearest/interpolation。
- per-rank asymmetric plan。
- SQLite store。
- trace replay routing。

## 17. 关键开放问题

1. `VirtualOp.formula` 是否直接存 flops/memory dict, 还是存 `RooflineInput` 对象。
2. operator schema contract 是否放在 `core/operator_schema`, collector 通过复制测试保持一致, 还是抽成可被 collector import 的独立小包。
3. attention 首版采 op-level 还是 module-level。
4. NCCL `algo/protocol` 首版是 predicted 还是 measured。
5. cudagraph mode 首版是否只用 `eager/cudagraph`, 还是直接细分 full/piecewise/fallback。
6. ModuleProfileBackend 何时引入, 是否等 OperatorDB GEMM/MoE 稳定后再做。

## 18. 最终形态

最终 LLMInferSim 的主路径应该是:

```text
adapter extracts workload
model template instantiates op graph
cost router annotates every op with measured/formula latency
trace explains source and gap
simulator sleeps/emits output according to trace
```

旧的 `layer_builder.py` 不再是演进中心, 只作为迁移期公式参考。
