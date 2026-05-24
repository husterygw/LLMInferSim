# Communication Subsystem Ideal Redesign Plan

## 1. 目标

这个文档描述 LLMInferSim 通信子系统的理想重构方案。这里不考虑旧代码兼容，不保留历史抽象包袱，目标是把通信建模纳入统一 Operator / Roofline / OperatorDB 体系。

核心目标：

- 通信操作也是普通 Operator，不再通过 `make_collective(...)` 这种弱语义 helper 构造。
- `CostRouter` 只决定 cost source，不关心 op 是计算还是通信。
- `RooflineBackend` 是唯一 roofline backend，内部根据 op 类型分发到 GEMM / Attention / MoE / Collective 的 roofline model。
- `OperatorDBBackend` 是唯一实测数据 backend，所有可签名 op 都通过 `op.signature()` 查询。
- 通信参数按 collective 类型组织，不再使用一个全局 `comm_step_latency` 解释所有通信。
- AllReduce / AllGather / ReduceScatter / AllToAll / P2P 分别建模，允许不同算法、协议、阈值和拓扑参数。
- 仿真运行时只使用参数化 roofline，不查询 collector JSONL，不做实测插值。

当前最迫切的问题是 TP=4 decode 小消息 AllReduce 被高估：Qwen dense decode 每 step 有 72 次 5KB 级 AllReduce，旧 tree 公式用 `2 * log2(n) * alpha`，导致单次约 40us；实测小消息 AllReduce 更像 `log2(n) * 7us`，最终 TPOT 系统性偏慢。

## 2. 总体架构

理想分层如下：

```text
ModelTemplate
  根据 model + workload + deploy 构造 Operator list

Operator
  描述 op 语义、shape、parallel、runtime
  提供 signature() 和 roofline_spec()

CostRouter
  只决定使用哪种 cost source:
    roofline_only
    operator_db_first
    require_operator_db

RooflineBackend
  唯一 roofline backend
  调用 op.roofline_spec()
  对 collective op 调用 collective roofline model

OperatorDBBackend
  唯一实测数据 backend
  调用 op.signature() 查 OperatorDB
```

关键边界：

- backend 表示 **cost 来源**，不是物理类别。
- 不应该有对外暴露的 `CommunicationRooflineBackend`。
- 不应该在 `CostRouter` 里写 `if op_kind == "collective"`。
- 通信模型是 `RooflineBackend` 的内部 estimator。

## 3. 目录结构

目标目录：

```text
llm_infer_sim/core/operators/
  collective.py             # Collective / AllReduce / AllGather / ReduceScatter / AllToAll / P2P
  gemm.py
  attention.py
  moe.py
  elementwise.py

llm_infer_sim/core/cost/backends/
  roofline.py               # 唯一 roofline backend
  operator_db.py            # 唯一实测数据 backend

llm_infer_sim/core/cost/roofline/
  communication.py          # collective roofline model
  compute.py                # 可选: compute roofline shared helpers

llm_infer_sim/core/profiles/
  hardware.py               # hardware + communication params
```

这里不是新增一套 `cost` 目录，而是在现有 `core/cost/` 内把职责收敛清楚。

## 4. 通信 Operator 设计

通信 op 直接建成具体类。

```python
@dataclass(frozen=True)
class Collective(Operator):
    name: str
    message_bytes: int
    world_size: int
    phase: str
    layer_idx: int | None
    ctx: OperatorContext
    group: str = "tp"
    topology_hint: str = ""
    comm_backend: str = "nccl"
    algorithm_hint: str | None = None
    protocol_hint: str | None = None

@dataclass(frozen=True)
class AllReduce(Collective):
    op_kind: str = "collective"
    op_subtype: str = "allreduce"

@dataclass(frozen=True)
class AllGather(Collective):
    op_kind: str = "collective"
    op_subtype: str = "allgather"

@dataclass(frozen=True)
class ReduceScatter(Collective):
    op_kind: str = "collective"
    op_subtype: str = "reducescatter"

@dataclass(frozen=True)
class AllToAll(Collective):
    op_kind: str = "collective"
    op_subtype: str = "alltoall"

@dataclass(frozen=True)
class P2P(Collective):
    op_kind: str = "collective"
    op_subtype: str = "p2p"
```

模型构造里直接写：

```python
AllReduce(
    name="tp_o_proj_allreduce",
    message_bytes=step.total_tokens * model.hidden_dim * ctx.a_byte,
    world_size=ctx.tp_size,
    phase=step.phase,
    layer_idx=layer_idx,
    ctx=ctx,
    group="tp",
    topology_hint=ctx.deploy.topology_hint,
)
```

不再使用：

```python
make_collective(...)
```

## 5. Operator API

所有 Operator 保持统一接口：

```python
class Operator:
    def shape(self) -> dict: ...
    def parallel(self) -> dict: ...
    def runtime(self) -> dict: ...
    def signature(self) -> OperatorSignature: ...
    def roofline_spec(self) -> RooflineSpec: ...
```

对于通信 op：

- `signature()` 用于 OperatorDB 对齐，包含 `op_kind/op_subtype/message_bytes/world_size/backend/topology/runtime`。
- `roofline_spec()` 返回 collective roofline 所需信息，不直接算时间。

示意：

```python
@dataclass(frozen=True)
class CollectiveRooflineSpec:
    comm_type: str
    message_bytes: int
    world_size: int
    backend: str
    topology_hint: str
    execution_mode: str
    algorithm_hint: str | None = None
    protocol_hint: str | None = None
```

Operator 负责描述自己，CostBackend 负责估算代价。

## 6. 通信参数设计

理想设计中不再使用全局 `comm_step_latency` 作为核心参数。

通信参数按 fabric、backend、collective 三层组织：

```python
@dataclass(frozen=True)
class CommunicationProfile:
    fabric: FabricProfile
    backends: dict[str, BackendCommunicationProfile]

@dataclass(frozen=True)
class FabricProfile:
    intra_node_links: dict[str, LinkProfile]
    inter_node_links: dict[str, LinkProfile]

@dataclass(frozen=True)
class LinkProfile:
    bandwidth_Bps: float
    startup_alpha_s: float
    topology: str

@dataclass(frozen=True)
class BackendCommunicationProfile:
    p2p: P2PParams
    allreduce: AllReduceParams
    allgather: AllGatherParams
    reducescatter: ReduceScatterParams
    alltoall: AllToAllParams
```

AllReduce 参数：

```python
@dataclass(frozen=True)
class AllReduceParams:
    small_tree_alpha_s: float
    small_tree_max_bytes: int
    small_tree_beta_scale: float

    ring_startup_alpha_s: float
    ring_beta_scale: float

    tree_startup_alpha_s: float
    tree_beta_scale: float

    protocol_efficiency: dict[str, float]
    algorithm_bias: dict[str, float]
```

P2P 参数：

```python
@dataclass(frozen=True)
class P2PParams:
    startup_alpha_s: float
    beta_scale: float
```

也就是说：

```text
p2p.startup_alpha_s
allreduce.small_tree_alpha_s
allreduce.ring_startup_alpha_s
alltoall.startup_alpha_s
```

是不同参数，不共享一个“万能 alpha”。

RTX 4090 首版建议：

```python
communication = CommunicationProfile(
    backends={
        "nccl": BackendCommunicationProfile(
            p2p=P2PParams(
                startup_alpha_s=9.6e-6,
                beta_scale=1.0,
            ),
            allreduce=AllReduceParams(
                small_tree_alpha_s=7.0e-6,
                small_tree_max_bytes=16 * 1024,
                small_tree_beta_scale=1.0,
                ring_startup_alpha_s=9.6e-6,
                ring_beta_scale=0.625,
                tree_startup_alpha_s=9.6e-6,
                tree_beta_scale=0.625,
                protocol_efficiency={
                    "simple": 0.625,
                    "ll": 0.8,
                    "ll128": 0.75,
                },
                algorithm_bias={},
            ),
            ...
        )
    }
)
```

这些数值来自离线校准，但仿真运行时只读参数，不查原始实测表。

## 7. AllReduce Roofline 模型

### 7.1 输入语义

```text
message_bytes = 每个 rank 的输入/输出 tensor bytes
world_size    = collective group size
```

Qwen dense TP：

```text
message_bytes = total_tokens * hidden_dim * activation_bytes
world_size = tp_size
```

### 7.2 候选算法

AllReduce 生成多个候选：

```text
small_tree
ring
large_tree
nvls        # 仅 NVSwitch/NVLink SHARP 类硬件启用
```

#### small_tree

小消息 latency-optimized 路径。

适用：

```text
message_bytes <= allreduce.small_tree_max_bytes
```

公式：

```text
depth = ceil(log2(world_size))

T_small_tree =
    depth * small_tree_alpha_s
  + message_bytes / beta_small
```

其中：

```text
beta_small = fabric_bandwidth * small_tree_beta_scale
```

这个模型用来表达 NCCL/pynccl 小消息行为：

```text
n=2: 约 1 hop
n=4: 约 2 hops
n=8: 约 3 hops
```

#### ring

大消息 bandwidth-oriented 路径。

公式：

```text
T_ring =
    2 * (world_size - 1) * ring_startup_alpha_s
  + 2 * (world_size - 1) / world_size * message_bytes / beta_ring
```

其中：

```text
beta_ring = fabric_bandwidth * ring_beta_scale
```

#### large_tree

保守 reduce+broadcast tree。

公式：

```text
depth = ceil(log2(world_size))

T_large_tree =
    2 * depth * tree_startup_alpha_s
  + 2 * message_bytes / beta_tree
```

它不是小消息 NCCL path，而是通用 fallback。

#### nvls

只在硬件 profile 显式声明时启用。

### 7.3 选择逻辑

```python
candidates = []

if message_bytes <= params.small_tree_max_bytes:
    candidates.append(small_tree)

candidates.append(ring)
candidates.append(large_tree)

if fabric.supports_nvls:
    candidates.append(nvls)

selected = min(candidate.time_s * algorithm_bias[candidate.name])
```

如果用户设置 `algorithm_hint`，则只允许匹配该算法；不匹配时明确报错或返回 unsupported，而不是静默 fallback。

### 7.4 eager / cudagraph

通信 algorithm time 和框架提交 overhead 分开：

```text
T_total = T_algorithm + T_runtime_submit
```

首版：

```text
cudagraph: T_runtime_submit = 0
eager:     T_runtime_submit = backend/runtime profile 中的 launch/sync overhead
```

不要把 eager overhead 合进 `small_tree_alpha_s`，否则同一组参数无法同时解释 eager 和 cudagraph。

## 8. 其他 Collective 模型

### 8.1 AllGather

候选：

```text
small_recursive_doubling
ring_allgather
```

公式方向：

```text
T_small = ceil(log2(n)) * small_alpha + bytes / beta_small
T_ring  = (n - 1) * alpha + (n - 1) / n * bytes / beta
```

### 8.2 ReduceScatter

候选：

```text
small_tree
ring_reducescatter
```

公式方向：

```text
T_ring = (n - 1) * alpha + (n - 1) / n * bytes / beta
```

### 8.3 AllToAll

AllToAll 不套 AllReduce 模型。

候选：

```text
pairwise_exchange
batched_pairwise
hierarchical_alltoall
```

首版：

```text
T_pairwise =
    (n - 1) * alltoall.startup_alpha_s
  + (n - 1) / n * bytes / beta_pairwise
  + contention_penalty
```

MoE EP 后续重点校准这一块。

### 8.4 P2P

```text
T_p2p = p2p.startup_alpha_s + bytes / beta_p2p
```

P2P 参数与 AllReduce 参数彻底解耦。

## 9. Trace 输出

Collective 的 `CostTraceEntry.metadata` 至少包含：

```python
{
    "comm_type": "allreduce",
    "message_bytes": 5120,
    "world_size": 4,
    "group": "tp",
    "backend": "nccl",
    "topology_hint": "concentrated",
    "execution_mode": "cudagraph",

    "selected_algorithm": "small_tree",
    "selected_protocol": "ll",
    "candidate_times_us": {
        "small_tree": 23.1,
        "ring": 41.8,
        "large_tree": 42.0,
    },

    "alpha_us": 7.0,
    "beta_GBps": 32.0,
    "latency_term_us": 14.0,
    "bandwidth_term_us": 9.1,
    "runtime_submit_overhead_us": 0.0,
}
```

这样 TP case 可以直接解释：

```text
TP=4 decode:
  72 个 allreduce
  每个约 20-25us
  总通信约 1.5-1.8ms

TP=4 prefill:
  72 个 allreduce
  每个约 3.2ms
  总通信约 230ms
```

## 10. 实施计划

### Step 1: 通信 Operator 具体化

改动：

- 新建/重写 `core/operators/collective.py`。
- 定义 `Collective / AllReduce / AllGather / ReduceScatter / AllToAll / P2P`。
- 删除生产路径中的 `make_collective(...)`。
- Qwen TP 直接构造 `AllReduce(...)`。

验收：

```bash
rg "make_collective" llm_infer_sim/core/models llm_infer_sim/core/operators
conda run -n llm_sim pytest tests/core/operators tests/core/models -q
```

### Step 2: 通信参数结构重做

改动：

- 在 hardware/profile 中新增 `CommunicationProfile`。
- 移除新路径对 `comm_step_latency` 的依赖。
- RTX 4090 写入 `p2p` 和 `allreduce` 首版参数。

验收：

```text
AllReduce 小消息使用 allreduce.small_tree_alpha_s
P2P 使用 p2p.startup_alpha_s
二者不是同一个字段
```

### Step 3: RooflineBackend 统一处理 Collective

改动：

- 删除对外的 `CommunicationRooflineBackend`。
- `CostRouter` 删除 collective 特判。
- `RooflineBackend.estimate(op)` 内部分发到 `_estimate_collective(op)`。
- `_estimate_collective(op)` 调用 `core/cost/roofline/communication.py`。

验收：

```bash
rg "CommunicationRooflineBackend|op_kind == .collective.|op.op_kind == .collective." llm_infer_sim/core/cost
conda run -n llm_sim pytest tests/core/cost -q
```

允许 `RooflineBackend` 内部有 collective dispatch。

### Step 4: AllReduce 新模型落地

改动：

- 实现 `small_tree / ring / large_tree / nvls` candidate。
- 支持 `algorithm_hint`。
- trace 输出所有候选时间。
- 小消息阈值默认来自 `AllReduceParams.small_tree_max_bytes`。

验收：

```text
n=2, 1KB: 选择 small_tree, latency 约 7-10us
n=4, 5KB: 选择 small_tree, latency 明显低于旧 40us
n=8, 1KB: latency 约 3 * small_tree_alpha
n=4, 10MB: 选择 ring 或大消息路径，不选 small_tree
```

### Step 5: Qwen dense TP 回归

运行：

```bash
bash scripts/run_bench_suite.sh tp_comm_sweep --filter-case '*prefill_i2048_o128__tp4'
bash scripts/run_bench_suite.sh tp_comm_sweep --filter-case '*tp2*'
bash scripts/run_bench_suite.sh tp_comm_sweep --filter-case '*tp4*'
```

预期：

```text
TP=4 prefill_i2048_o128 TTFT 继续接近实测
TP=4 prefill_i2048_o128 TPOT 从 +31% 明显收敛
TP=2 不明显退化
```

### Step 6: AllToAll / EP 后续展开

在 TP AllReduce 稳定后，再做：

- MoE EP dispatch/combine 的 `AllToAll`。
- AllToAll pairwise/hierarchical model。
- EP topology / load imbalance / routing distribution。

不和 TP AllReduce 同时做，避免误差来源混在一起。

## 11. 测试计划

单元测试：

```bash
conda run -n llm_sim pytest tests/core/operators -q
conda run -n llm_sim pytest tests/core/cost -q
conda run -n llm_sim pytest tests/core/models -q
```

Benchmark smoke：

```bash
bash scripts/run_bench_suite.sh tp_comm_sweep --filter-case '*tp2*prefill_i2048_o128*'
bash scripts/run_bench_suite.sh tp_comm_sweep --filter-case '*tp4*prefill_i2048_o128*'
```

完整 TP suite：

```bash
bash scripts/run_bench_suite.sh tp_comm_sweep
bash scripts/run_bench_suite.sh tp_batch_sweep
```

## 12. 验收标准

架构验收：

```text
模型模板直接构造 AllReduce / AllToAll 等具体通信 Operator
生产路径不再使用 make_collective
CostRouter 不再按 op_kind 特判 communication
RooflineBackend 是唯一 roofline backend
OperatorDBBackend 是唯一 measured DB backend
通信参数不再依赖全局 comm_step_latency
AllReduce / P2P / AllToAll 参数互相独立
```

数值验收：

```text
TP=4 prefill_i2048_o128 TTFT gap 保持在 ±5% 左右
TP=4 prefill_i2048_o128 TPOT gap 从 +31% 收敛到 +15% 内
TP=2 aggregate 不退化到 FAIL
小消息 AllReduce trace 显示 selected_algorithm=small_tree
大消息 AllReduce trace 不走 small_tree
```

建模验收：

```text
运行时不查 collector JSONL
运行时不做 measured interpolation
实测数据只用于离线校准参数
trace 能解释 alpha / beta / candidate / selected algorithm
```

## 13. 不做的事情

第一轮不做：

```text
不接 collective OperatorDB exact hit
不做实测插值
不做通信 overlap
不做跨节点 hierarchical allreduce
不一次性校准 AllGather / ReduceScatter / AllToAll
不把 MoE EP 和 TP AllReduce 混在同一轮改动里
```

先把 Qwen dense TP AllReduce 做干净，再展开 EP / MoE / 跨节点。
