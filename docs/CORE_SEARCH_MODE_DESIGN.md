# LLMInferSim Core Search Mode 设计方案

本文档补充 `LLMINFERSIM_SYSTEM_SOLUTION_V3.md` 和
`LLMINFERSIM_IMPLEMENTATION_PLAN.md`，重点回答一个问题：

> 做最佳部署空间搜索时，是否可以不依赖 vLLM scheduler，也不通过 `sleep` 推进时间？

结论：可以，而且应该这样做。

空间搜索需要快速评估大量配置，如果每个候选配置都接入真实 vLLM 并按真实时间等待，
搜索速度会接近真实线上运行，失去仿真的意义。LLMInferSim 应该提供一个 core-native
search mode：在 core 层实现轻量 scheduler、KV 容量账本、虚拟时间推进和指标聚合。
vLLM/sglang 等真实框架只用于采集、对齐和 top-K 配置验证。

当前执行边界:

```text
现在不实现 core_search。
现在只把单次仿真做成 search-ready。
```

也就是说, 当前主线只要求 `DeployConfig` 能贯穿:

```text
ModelGraphTemplate / OpFactory
VirtualOp.parallel/runtime
CostRouter / OperatorDB key
StepCostTrace / Report metadata
```

等 TTFT baseline、OperatorDB 和模型级对齐稳定后, 再按本文档实现真正的
`core_search` 外层循环。

---

## 1. 目标与非目标

### 1.1 目标

1. 支持部署空间搜索：
   - TP / PP / EP / DP
   - batch token 上限
   - max seqs
   - chunked prefill
   - KV cache block 数
   - eager / cudagraph execution mode
   - backend / kernel / operator DB 版本
2. 搜索时完全使用 simulator time：
   - 不调用 `time.sleep`
   - 不依赖真实 vLLM scheduler step
   - 一个候选配置可以用远快于真实时间的速度跑完 workload
3. 输出线上关心的请求级指标：
   - TTFT
   - TPOT
   - E2E latency
   - throughput
   - SLA violation
   - KV peak usage
   - step breakdown
4. 保持 core 框架无关：
   - core scheduler 不出现 vLLM 内部对象
   - vLLM/sglang adapter 只负责 trace extraction、collector、validation

### 1.2 非目标

首版不追求完全复刻 vLLM scheduler：

1. 不实现完整 preemption / recompute。
2. 不实现复杂 prefix cache eviction。
3. 不实现 speculative decoding。
4. 不实现完整多 engine 异步 pipeline。
5. 不保证 step 序列和 vLLM 完全一致。

core search mode 的目标是做配置排序和趋势判断。最终候选配置仍需要通过
framework live validation 校准。

---

## 2. 三种运行模式

LLMInferSim 最终应该有三种互补的运行模式。

### 2.1 `core_search`

用于部署空间搜索。

特点：

- core scheduler 自己生成 step。
- 使用 `VirtualClock` 推进 simulator time。
- 不 sleep。
- 不依赖 vLLM runtime。
- 速度最快。

适用场景：

- 大规模配置搜索。
- 对比 TP/EP/DP/PP。
- 对比 batch token、max seqs、chunked prefill 参数。
- 初步判断 eager/cudagraph 对指标的影响。

### 2.2 `trace_replay`

用于固定调度轨迹下的 cost what-if。

特点：

- 输入是真实 vLLM/sglang 导出的 schedule trace。
- 不重新调度，只重放每个 step 的 `GlobalStepWorkload`。
- 可以替换 cost backend、operator DB、hardware profile。
- 不 sleep。

适用场景：

- 同一个调度轨迹下比较不同硬件或 kernel profile。
- 分析 cost model 误差。
- 对齐真实框架的 step 形状分布。

### 2.3 `framework_live_validation`

用于 top-K 真实验证。

特点：

- 接入 vLLM/sglang runtime。
- 可以按真实时间推进，也可以使用已有的 time emulator。
- 收集真实 scheduler output、真实 step shape、真实指标。

适用场景：

- 验证 core search 选出的 top-K 配置。
- 发现 core scheduler 和真实框架 scheduler 的系统偏差。
- 生成新的 calibration / operator DB 数据。

---

## 3. 总体架构

核心闭环如下：

```text
RequestSource
  -> CoreScheduler
  -> GlobalStepWorkload
  -> ModelOpBuilder
  -> StepCostEngine
  -> VirtualClock
  -> RequestState Update
  -> MetricsCollector
  -> SearchRunner
```

每层职责：

| 模块 | 职责 |
| --- | --- |
| `RequestSource` | 产生请求流，包含 arrival time、prompt length、output length |
| `CoreScheduler` | 根据当前请求状态和调度配置生成下一个 step |
| `GlobalStepWorkload` | 框架无关 step workload，现有结构可以继续使用 |
| `ModelOpBuilder` | 把 step workload 实例化成 VirtualOp list |
| `StepCostEngine` | 查询 OperatorDB / ModuleProfile / Roofline，得到 step latency |
| `VirtualClock` | 用 step latency 推进 simulator time |
| `RequestState Update` | 更新 prefill/decode 进度、首 token 时间、完成时间 |
| `MetricsCollector` | 聚合 TTFT/TPOT/E2E/throughput/SLA |
| `SearchRunner` | 遍历候选配置，输出 Pareto frontier 和推荐配置 |

---

## 4. 建议目录结构

```text
llm_infer_sim/
  core/
    scheduler_sim/
      __init__.py
      request.py              # SearchRequest / SimRequestState
      config.py               # SchedulerConfig
      queue.py                # waiting/running/finished queues
      kv_capacity.py          # core-native KVCapacityState
      policy.py               # decode-first / prefill-first / mixed policy
      scheduler.py            # CoreScheduler
      chunked_prefill.py      # chunk 切分策略
      admission.py            # admit / reject / delay 逻辑

    simulation/
      virtual_clock.py        # 只维护 simulator time，不 sleep
      time_emulator.py        # 保留给 framework live mode

    replay/
      __init__.py
      schedule_trace.py       # trace schema
      trace_reader.py         # vLLM/sglang trace -> GlobalStepWorkload
      trace_runner.py         # 固定 step replay

    workload/
      workload.py             # 现有 GlobalStepWorkload 可复用
      request_state.py        # 现有 RequestMetrics 可复用

  search/
    __init__.py
    search_space.py           # 参数空间定义与展开
    runner.py                 # CoreSearchRunner
    pareto.py                 # Pareto frontier
    picking.py                # SLA-aware 推荐策略
    report.py                 # CSV/JSON/Markdown 报告

  adapters/
    vllm/
      trace_exporter.py       # live run -> schedule trace
      validation_runner.py    # top-K validation
    sglang/
      trace_exporter.py
      validation_runner.py
```

注意：

- `scheduler_sim` 是 core-native 调度器，不依赖 vLLM。
- `replay` 是固定 trace 回放，不做重新调度。
- `adapters/vllm` 和 `adapters/sglang` 只负责真实框架对接。
- 现有 `core/simulation/kv_block_allocator.py` 更适合作为 framework trace observer，
  不建议直接承担 core search 的容量约束。

---

## 5. 核心数据结构

### 5.1 请求输入

```python
@dataclass
class SearchRequest:
    request_id: str
    arrival_time: float
    prompt_len: int
    output_len: int
    priority: int = 0
    prefix_cache_key: str | None = None
```

`SearchRequest` 是 workload 输入，不记录执行进度。

### 5.2 请求状态

```python
class SimRequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    REJECTED = "rejected"


@dataclass
class SimRequestState:
    request: SearchRequest
    status: SimRequestStatus = SimRequestStatus.WAITING

    computed_prompt_tokens: int = 0
    generated_tokens: int = 0
    context_len: int = 0

    arrival_time: float = 0.0
    first_token_time: float | None = None
    completion_time: float | None = None

    last_decode_token_time: float | None = None
    per_token_latencies: list[float] = field(default_factory=list)
```

状态推进规则：

- `computed_prompt_tokens < prompt_len`：还在 prefill。
- `computed_prompt_tokens == prompt_len` 且 `generated_tokens < output_len`：decode。
- `generated_tokens == output_len`：finished。
- `context_len = computed_prompt_tokens + generated_tokens`。

### 5.3 调度配置

```python
@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int
    max_num_seqs: int

    block_size: int
    num_gpu_blocks: int
    kv_dtype_bytes: float = 2.0

    enable_chunked_prefill: bool = True
    max_num_partial_prefills: int = 1
    prefill_chunk_size: int = 2048

    policy: str = "decode_first_mixed"
    enable_prefix_cache: bool = False
    prefix_cache_hit_ratio: float = 0.0

    enable_preemption: bool = False
```

首版建议固定：

- `policy = "decode_first_mixed"`
- `enable_preemption = False`
- `enable_prefix_cache = False`

等基本 TTFT/TPOT 走通后，再逐步打开 prefix cache 和 preemption。

### 5.4 Core scheduler 输出

Core scheduler 直接输出现有的 `GlobalStepWorkload`：

```python
@dataclass
class GlobalStepWorkload:
    step_id: int
    phase: StepPhase
    requests: list[RequestWorkload]
    num_prefill_tokens: int
    num_decode_tokens: int
    total_scheduled_tokens: int
    num_prefill_requests: int
    num_decode_requests: int
    num_prefix_cached_tokens: int = 0
```

这样后续 `ModelOpBuilder -> StepCostEngine` 可以复用同一个 workload schema。

---

## 6. CoreScheduler 调度算法

首版用一个确定性的 mixed batching scheduler。

### 6.1 每步流程

```text
while not all requests finished:
  1. admit arrivals whose arrival_time <= virtual_clock.now
  2. free KV blocks of finished requests
  3. schedule decode tokens for running requests
  4. schedule prefill chunks for waiting/prefilling requests
  5. check token / seq / KV capacity limits
  6. produce GlobalStepWorkload
  7. estimate step latency
  8. virtual_clock.advance(step_latency)
  9. update request states and metrics
```

如果当前没有可调度请求，但未来还有请求到达：

```text
virtual_clock.jump_to(next_arrival_time)
```

这里是时间跳跃，不是 sleep。

### 6.2 Decode-first mixed policy

建议首版使用 decode-first mixed policy，因为它更贴近 serving 系统里保护 TPOT 的常见策略。

规则：

1. 先给所有 `DECODING` 请求各分配 1 个 decode token。
2. 如果没有超过 `max_num_seqs` 和 `max_num_batched_tokens`，剩余 token budget 给 prefill。
3. prefill 可以是完整 prompt，也可以是 chunked prefill。
4. 如果 prefill 加入后超过 KV 容量，则减少或延后 prefill 请求。
5. 如果 decode 本身超过容量，首版可以阻塞新 prefill，不做 preemption。

伪代码：

```python
def schedule_next_step(state, config, now):
    admit_new_arrivals(state, now)

    token_budget = config.max_num_batched_tokens
    seq_budget = config.max_num_seqs
    scheduled = []

    # 1. decode first
    for req in state.decoding_requests():
        if token_budget <= 0 or seq_budget <= 0:
            break
        if not state.kv.can_extend(req, extra_tokens=1):
            continue
        scheduled.append(decode_workload(req, num_tokens=1))
        token_budget -= 1
        seq_budget -= 1

    # 2. prefill with remaining budget
    partial_prefills = 0
    for req in state.prefill_candidates():
        if token_budget <= 0 or seq_budget <= 0:
            break
        if partial_prefills >= config.max_num_partial_prefills:
            break

        remain = req.prompt_len - req.computed_prompt_tokens
        chunk = min(remain, token_budget)
        if config.enable_chunked_prefill:
            chunk = min(chunk, config.prefill_chunk_size)

        chunk = state.kv.clip_to_capacity(req, chunk)
        if chunk <= 0:
            continue

        scheduled.append(prefill_workload(req, num_tokens=chunk))
        token_budget -= chunk
        seq_budget -= 1
        if chunk < remain:
            partial_prefills += 1

    return build_global_step_workload(scheduled)
```

### 6.3 Step phase 判定

```text
only prefill          -> PREFILL
only decode           -> DECODE
prefill + decode      -> MIXED
only chunked prefill  -> CHUNKED_PREFILL
```

如果 mixed step 中包含 chunked prefill，仍建议标为 `MIXED`，并在每个
`RequestWorkload.is_chunked` 上保留细节。

---

## 7. KV 容量建模

空间搜索必须有 core-native KV 容量约束，否则 batch 参数会不真实。

### 7.1 为什么不能直接用现有 KVBlockAllocator

现有 `core/simulation/kv_block_allocator.py` 主要观察 vLLM scheduler output：

- 输入是 `scheduled_new_reqs`
- 输入是 `scheduled_cached_reqs`
- 输入是 `finished_req_ids`
- 适合真实框架 trace 统计

core search 需要在调度前判断容量：

- 某个 prefill chunk 能不能放进去？
- decode 多 1 token 会不会跨 block？
- 是否能 admit 新请求？
- 如果不能，延后还是 preempt？

所以需要新建 `KVCapacityState`。

### 7.2 KVCapacityState

```python
@dataclass
class KVCapacityState:
    block_size: int
    num_blocks_total: int
    block_bytes: int
    req_blocks: dict[str, int] = field(default_factory=dict)

    def blocks_for_tokens(self, tokens: int) -> int: ...
    def current_blocks(self, req_id: str) -> int: ...
    def can_reserve(self, req_id: str, target_context_len: int) -> bool: ...
    def reserve(self, req_id: str, target_context_len: int) -> None: ...
    def release(self, req_id: str) -> None: ...
    def available_blocks(self) -> int: ...
```

block bytes 公式可以复用现有 `compute_block_bytes`：

- MLA: `(kv_lora_rank + rope_head_dim) * kv_byte`
- MHA/GQA: `num_kv_heads * head_dim * 2 * kv_byte`

### 7.3 首版容量策略

首版只做 admission control：

1. prefill 请求如果容量不足，就延后。
2. decode 请求如果容量不足，停止 admit 新 prefill。
3. 不做 preemption。
4. 不做 eviction。
5. 不做 prefix cache sharing。

这已经足够让空间搜索感受到：

- KV blocks 越少，可并发 seq 越少。
- prompt 越长，占用越大。
- chunked prefill 会改变入队和首 token 行为。

---

## 8. StepCostEngine 接入

Core scheduler 只产生 workload，不关心算子时间。

每个 step 的 cost 仍走新架构中的标准路径：

```text
GlobalStepWorkload
  -> ModelOpBuilder
  -> list[VirtualOp]
  -> CostRouter
  -> OperatorDB / ModuleProfile / Roofline
  -> TimelineBackend
  -> StepCostResult
```

### 8.1 StepCostResult

```python
@dataclass
class StepCostResult:
    total_latency_s: float
    device_compute_s: float
    communication_s: float
    scheduler_overhead_s: float = 0.0
    timeline_overhead_s: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    cost_sources: dict[str, str] = field(default_factory=dict)
```

`total_latency_s` 用于推进 `VirtualClock`。

### 8.2 eager / cudagraph 在哪里体现

不要在 `CoreScheduler` 里体现 eager/cudagraph。

正确位置是 `TimelineBackend`：

- `CoreScheduler`：决定本 step 有哪些 token。
- `ModelOpBuilder`：决定本 step 有哪些 VirtualOp。
- `CostRouter`：决定每个 op 的 device time。
- `TimelineBackend`：决定 eager launch pipeline 或 cudagraph replay 后的 step latency。

即：

```text
same workload + same ops
different execution_mode
  -> different timeline model
  -> different StepCostResult.total_latency_s
```

首版可以先不实现完整 eager pipeline，只保留字段：

```python
execution_mode: Literal["eager", "cudagraph"]
```

TTFT baseline 走通后，再加入：

- eager launch pipeline
- cudagraph capture size padding
- cudagraph replay overhead

---

## 9. VirtualClock

现有 `VirtualTimeEmulator(mode="instant")` 的语义是“不 sleep”，但它没有保存当前时间。
core search 需要一个明确的 clock：

```python
class VirtualClock:
    def __init__(self) -> None:
        self.now_s = 0.0

    def now(self) -> float:
        return self.now_s

    def advance(self, delta_s: float) -> None:
        self.now_s += max(delta_s, 0.0)

    def jump_to(self, target_s: float) -> None:
        self.now_s = max(self.now_s, target_s)
```

`VirtualTimeEmulator` 可以继续留给 live validation mode：

- `instant`：不 sleep
- `sleep`：真实等待

`VirtualClock` 则专门用于 core search 和 trace replay。

---

## 10. 指标更新

每个 step 完成后，用 `clock.now()` 更新请求状态。

### 10.1 Prefill 更新

```python
req.computed_prompt_tokens += scheduled_prefill_tokens
req.context_len += scheduled_prefill_tokens

if req.computed_prompt_tokens == req.prompt_len:
    req.status = DECODING
```

prefill 完成时还没有输出 token，因此不记录 TTFT。

### 10.2 Decode 更新

```python
req.generated_tokens += 1
req.context_len += 1

if req.first_token_time is None:
    req.first_token_time = clock.now()
    req.per_token_latencies.append(clock.now() - req.arrival_time)
else:
    req.per_token_latencies.append(clock.now() - req.last_decode_token_time)

req.last_decode_token_time = clock.now()
```

如果 `generated_tokens == output_len`：

```python
req.status = FINISHED
req.completion_time = clock.now()
kv.release(req.request_id)
```

### 10.3 复用现有 RequestMetrics

现有 `core/workload/request_state.py` 已经是 simulator-time，可以继续使用。

建议新增一个转换函数：

```python
def to_request_metrics(state: SimRequestState) -> RequestMetrics:
    ...
```

这样 core search、trace replay、framework validation 都能输出同一种 metrics。

---

## 11. SearchRunner

### 11.1 SearchSpace

```python
@dataclass
class DeploymentCandidate:
    name: str
    tp_size: int
    pp_size: int
    ep_size: int
    dp_size: int

    max_num_batched_tokens: int
    max_num_seqs: int
    block_size: int
    num_gpu_blocks: int

    enable_chunked_prefill: bool
    prefill_chunk_size: int
    execution_mode: str
    backend: str
    operator_db_tag: str
```

### 11.2 Runner 流程

```python
class CoreSearchRunner:
    def run(self, requests, candidates):
        results = []
        for candidate in candidates:
            model_profile = build_model_profile(candidate)
            deploy_profile = build_deploy_profile(candidate)
            cost_engine = build_cost_engine(model_profile, deploy_profile, candidate)

            scheduler = CoreScheduler(
                config=build_scheduler_config(candidate),
                model=model_profile,
            )
            clock = VirtualClock()

            result = run_one_candidate(
                requests=requests,
                scheduler=scheduler,
                cost_engine=cost_engine,
                clock=clock,
            )
            results.append(result)

        return rank_and_report(results)
```

### 11.3 CandidateResult

```python
@dataclass
class CandidateResult:
    candidate: DeploymentCandidate
    total_sim_time_s: float
    completed_requests: int
    rejected_requests: int

    avg_ttft_s: float
    p90_ttft_s: float
    p99_ttft_s: float
    avg_tpot_s: float
    p90_tpot_s: float
    p99_tpot_s: float
    avg_e2e_s: float
    p90_e2e_s: float
    p99_e2e_s: float

    output_tokens_per_s: float
    requests_per_s: float
    peak_kv_blocks: int
    peak_kv_utilization: float

    avg_step_breakdown: dict[str, float]
    cost_source_summary: dict[str, int]
```

### 11.4 排序策略

建议输出两种结果：

1. Pareto frontier：
   - maximize throughput
   - minimize p99 TTFT
   - minimize p99 TPOT
   - minimize GPU count
2. SLA-aware best pick：
   - 先过滤 `p99_ttft <= target_ttft`
   - 再过滤 `p99_tpot <= target_tpot`
   - 再选择 GPU 数最少或 throughput 最高的配置

---

## 12. 与 vLLM 的关系

core search 不依赖 vLLM scheduler，但不意味着 vLLM 没用。

推荐工作流：

```text
1. collector 收集算子级数据
2. operator_db 建库
3. core_search 大规模搜索候选部署
4. 选出 top-K
5. framework_live_validation 跑 vLLM/sglang
6. 对比真实指标与 core search 预测
7. 修正 scheduler policy / overhead / operator_db
```

这样分工最清晰：

| 层级 | 用途 |
| --- | --- |
| Collector | 收集底层算子真实性能 |
| OperatorDB | 保存 shape/backend/hardware 下的 latency |
| CoreScheduler | 快速生成假想部署下的 step 序列 |
| StepCostEngine | 估算每个 step 的耗时 |
| vLLM/sglang live | 只验证 top-K 和校准误差 |

---

## 13. 实施步骤

### Step 1: 建立 core search 基础结构

新增：

```text
core/scheduler_sim/request.py
core/scheduler_sim/config.py
core/scheduler_sim/scheduler.py
core/simulation/virtual_clock.py
search/search_space.py
search/runner.py
```

完成：

- `SearchRequest`
- `SimRequestState`
- `SchedulerConfig`
- `VirtualClock`
- 最小 `CoreScheduler`
- 最小 `CoreSearchRunner`

不做：

- KV 容量
- chunked prefill
- prefix cache
- cudagraph timeline

### Step 2: 跑通 prefill -> decode 生命周期

要求：

- 请求按 arrival time 进入系统。
- prefill 完成后进入 decode。
- decode 每 step 生成 1 token。
- 请求完成后记录 completion time。
- 输出 TTFT/TPOT/E2E。

此阶段 cost 可以先用一个 mock cost engine：

```python
latency = a * prefill_tokens + b * decode_tokens + c
```

这样先验证 scheduler 和 metrics。

### Step 3: 接入真实 StepCostEngine

把 `GlobalStepWorkload` 接到：

```text
ModelOpBuilder -> VirtualOp -> CostRouter -> StepCostResult
```

要求：

- 同一个 `GlobalStepWorkload` 可同时用于 core search 和 trace replay。
- cost source 能区分 `operator_db` / `module_profile` / `roofline`。
- result 中保留 step breakdown。

### Step 4: 加入 KVCapacityState

新增：

```text
core/scheduler_sim/kv_capacity.py
```

完成：

- block_size / num_gpu_blocks 约束。
- prefill 前检查容量。
- decode 跨 block 时扩容。
- finished 后释放 KV。
- 记录 peak KV utilization。

首版仍不做 preemption。

### Step 5: 加入 chunked prefill

新增：

```text
core/scheduler_sim/chunked_prefill.py
```

完成：

- `enable_chunked_prefill`
- `prefill_chunk_size`
- `max_num_partial_prefills`
- mixed step 中 prefill + decode 共存

重点验证：

- chunk 太小 TTFT 变好但总体 throughput 可能下降。
- chunk 太大 decode 被阻塞，TPOT 变差。

### Step 6: 加入搜索与报告

新增：

```text
search/pareto.py
search/picking.py
search/report.py
```

完成：

- grid search / yaml search space。
- CSV/JSON/Markdown 输出。
- Pareto frontier。
- SLA-aware 推荐。

### Step 7: 加入 trace replay

新增：

```text
core/replay/schedule_trace.py
core/replay/trace_reader.py
core/replay/trace_runner.py
```

完成：

- 从 vLLM adapter 导出的 trace 读取 `GlobalStepWorkload`。
- 固定 step 序列重放。
- 与 core scheduler 结果对比 shape distribution。

### Step 8: top-K live validation

新增：

```text
adapters/vllm/validation_runner.py
adapters/sglang/validation_runner.py
```

完成：

- 对 core search top-K 配置跑真实框架。
- 比较预测 TTFT/TPOT/E2E 与真实指标。
- 生成 calibration report。

---

## 14. 测试用例

### 14.1 Scheduler 单测

1. `test_arrival_admission`
   - arrival time 未到的请求不进入调度。
2. `test_prefill_then_decode`
   - prefill 完成后进入 decode。
3. `test_decode_one_token_per_step`
   - decode 请求每步生成 1 token。
4. `test_max_num_batched_tokens`
   - 每步 token 总数不超过配置。
5. `test_max_num_seqs`
   - 每步请求数不超过配置。
6. `test_idle_jump_to_next_arrival`
   - 空闲时 clock 跳到下个 arrival，不 sleep。

### 14.2 KV 单测

1. `test_kv_reserve_and_release`
2. `test_decode_cross_block_boundary`
3. `test_prefill_delayed_when_capacity_insufficient`
4. `test_peak_kv_utilization`

### 14.3 Chunked prefill 单测

1. `test_prefill_split_into_chunks`
2. `test_mixed_decode_and_prefill`
3. `test_max_num_partial_prefills`

### 14.4 Metrics 单测

1. `test_ttft`
2. `test_tpot`
3. `test_e2e_latency`
4. `test_throughput`

### 14.5 Search 单测

1. `test_search_runs_multiple_candidates`
2. `test_pareto_frontier`
3. `test_sla_picking`
4. `test_report_schema`

---

## 15. 首版最小可用方案

真正进入搜索阶段时, 建议 MVP 只做这些：

```text
1. SearchRequest
2. SimRequestState
3. SchedulerConfig
4. VirtualClock
5. CoreScheduler(decode-first mixed, no KV)
6. MockCostEngine
7. RequestMetrics 输出
8. CoreSearchRunner 跑多个 candidate
```

然后按顺序补：

```text
9. 接入真实 StepCostEngine
10. KVCapacityState
11. chunked prefill
12. Pareto report
13. trace replay
14. vLLM top-K validation
```

这个顺序的好处是：

- 先验证时间推进和请求生命周期。
- 再接入算子 cost。
- 再加入 KV 和 chunked prefill 等调度复杂度。
- 最后才做真实框架对齐。

---

## 16. 关键设计判断

1. 空间搜索不应该依赖 vLLM scheduler。
   vLLM scheduler 是验证对象之一，不应该成为搜索内循环的硬依赖。

2. core scheduler 不追求完全复刻真实框架。
   它要提供稳定、可解释、可快速搜索的调度模型。

3. `sleep` 只应该出现在 live validation。
   core search 和 trace replay 都必须使用虚拟时间。

4. scheduler 和 cost model 必须解耦。
   scheduler 只决定 step shape，cost model 决定 step latency。

5. eager/cudagraph 不应该写进 scheduler。
   它属于 timeline backend，同一组 ops 在不同 execution mode 下得到不同 latency。

6. KV 容量必须在 core 层实现。
   否则部署搜索无法正确比较 batch、seq、context length 和 GPU memory。

7. vLLM/sglang 的价值在 collector 和 validation。
   算子数据来自真实框架，但 core search 的对象应该是框架无关的 VirtualOp 和 workload。
