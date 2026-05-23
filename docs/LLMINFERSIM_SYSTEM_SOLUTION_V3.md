# LLMInferSim 系统方案

> 状态: current system solution  
> 日期: 2026-05-23  
> 说明: 本文档取代 2026-05-19 的 V3 draft。当前主线已从 `VirtualOp + OpFactory + StepOpPlan` 收敛为 `Operator + GroupedStepPlan + CostRouter`。

## 1. 系统定位

LLMInferSim 是面向 LLM serving 的可解释性能仿真系统。

它的定位不是重新实现一个外部 scheduler, 也不是简单复刻 AIConfigurator 的静态估算, 而是:

```text
保留真实推理框架的 scheduler / request lifecycle / KV cache 行为,
只替换 model execution 的耗时与输出。
```

当前主路径:

```text
Framework Adapter
  -> GlobalStepWorkload / StepShape
  -> ModelTemplate
  -> GroupedStepPlan[Operator]
  -> CostRouter
  -> StepCostTrace
  -> Virtual execution / fake output
  -> Framework scheduler continues
```

系统核心价值:

- 真实接入 vLLM scheduler, 能暴露真实调度和软件路径问题。
- core 层保持框架无关, 未来可接 SGLang。
- 通过 OperatorDB 对齐 collector 实测算子数据。
- 通过 Roofline 给出可解释 lower bound 和 gap 归因。
- 通过 case-driven benchmark 做模型级 TTFT/TPOT 校准。

## 2. 参考系统吸收边界

### 2.1 从 AIConfigurator 吸收

吸收:

- `Model -> Operation list` 的模型表达思想。
- typed operator database。
- operator collector / microbench。
- `kernel_source / framework_version / execution_mode` 进入数据 key。
- MoE routing distribution / power-law correction 的思想。
- 配置搜索可以作为后期外层能力。

不照搬:

- 不把模型和单一框架 kernel 绑定死。
- 不以静态 steady-state envelope 作为唯一主路径。
- 不让 op 自己决定 cost policy。

### 2.2 从 LLMServingSim 吸收

吸收:

- request-level workload / metrics。
- virtual time / fast simulation 思路。
- mixed prefill/decode shape 的表达。
- communication event 与拓扑建模思想。

不照搬:

- 不在 vLLM 外部重写完整 scheduler。
- 不把 module profile 作为首轮主路径。

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
│ VllmAdapter / VirtualWorker / VirtualModelRunner             │
│ framework object -> GlobalStepWorkload / StepShape           │
└──────────────────────────────┬──────────────────────────────┘
                               │ StepShape
┌──────────────────────────────▼──────────────────────────────┐
│ Model 层                                                     │
│ QwenModelTemplate / DeepSeekModelTemplate                    │
│ StepShape + ModelConfig + DeployConfig -> GroupedStepPlan    │
└──────────────────────────────┬──────────────────────────────┘
                               │ GroupedStepPlan[Operator]
┌──────────────────────────────▼──────────────────────────────┐
│ Cost 层                                                      │
│ CostRouter / OperatorDBBackend / RooflineBackend             │
│ Operator -> CostTraceEntry -> StepCostTrace                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ StepCostTrace
┌──────────────────────────────▼──────────────────────────────┐
│ Simulation / Metrics 层                                      │
│ Time emulator / fake output / metrics reporter               │
│ 输出 TTFT / TPOT / E2E / throughput / breakdown              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 离线数据采集与校准平面

```text
┌─────────────────────────────────────────────────────────────┐
│ Collector / Profiler                                        │
│ operator microbench / collective bench / model benchmark     │
└──────────────────────────────┬──────────────────────────────┘
                               │ RawRecord JSONL
┌──────────────────────────────▼──────────────────────────────┐
│ Importer / Canonicalizer                                    │
│ collector_v2 importer / OperatorSignature canonicalizer      │
└──────────────────────────────┬──────────────────────────────┘
                               │ OperatorRecord
┌──────────────────────────────▼──────────────────────────────┐
│ Data Asset 层                                                │
│ OperatorDB JSONL / MemoryStore / future store                │
└──────────────────────────────┬──────────────────────────────┘
                               │ queryable measured records
┌──────────────────────────────▼──────────────────────────────┐
│ Regression / Report                                         │
│ roofline gap / coverage / model-level real-vs-sim report     │
└─────────────────────────────────────────────────────────────┘
```

## 4. 最终能力与功能数据流

这一节面向“给别人介绍这个工程”的视角: 先说明系统最终提供什么能力, 以及每个能力的数据如何流动。后面的核心抽象章节再解释这些能力如何实现。

### 4.1 能力总览

LLMInferSim 最终提供五类能力:

```text
1. 模型级性能仿真
   输入 workload / deploy / model, 输出 TTFT / TPOT / E2E / throughput。

2. 算子级 roofline vs 实测对比
   输入 collector operator DB, 输出 roofline lower bound、实测 latency、gap。

3. 模型级 real-vs-sim 校准
   输入 benchmark suite, 同时跑真实 vLLM 与 virtual backend, 输出 gap report。

4. 调试与归因
   将模型级误差拆到 operator、phase、memory/compute/communication/source。

5. 部署配置搜索准备
   用统一 DeployConfig 参数化单次仿真, 后续可在外层枚举候选。
```

### 4.2 功能一: 模型级性能仿真

目标: 在真实 vLLM scheduler 里替换模型执行耗时, 快速得到请求级指标。

数据流:

```text
用户请求 / vLLM scheduler
  -> Adapter 提取当前 step
  -> GlobalStepWorkload / StepShape
  -> ModelTemplate.build_grouped_step()
  -> GroupedStepPlan[Operator]
  -> CostRouter
       -> OperatorDBBackend 或 RooflineBackend
  -> StepCostTrace
  -> TimeEmulator / fake output
  -> vLLM scheduler 继续推进
  -> TTFT / TPOT / E2E / throughput
```

输出:

```text
request-level metrics
step-level latency
operator/grouped trace
source breakdown: operator_db / roofline / skipped
compute vs memory bottleneck
```

用途:

- 在不真实执行模型计算的情况下跑 vLLM request lifecycle。
- 看部署参数对 TTFT/TPOT 的影响。
- 快速定位模型级耗时主要来自 prefill、decode、通信还是 runtime。

### 4.3 功能二: 算子级 roofline vs 实测对比

目标: 用 collector 采集的 operator latency 检查 roofline 建模是否合理。

数据流:

```text
collector raw jsonl
  -> collector_v2 importer
  -> OperatorRecord
  -> OperatorSignature
  -> report_operator_roofline_gap.py
       -> 从 signature 还原 Operator
       -> op.roofline_spec()
       -> RooflineBackend.estimate()
  -> per-record gap report
```

输出:

```text
op_kind / op_subtype
shape / dtype / tp / execution_mode
measured_us_p50 / p10 / p90
roofline_us
roofline_gap = measured_us_p50 / roofline_us
bottleneck
arithmetic_intensity
```

用途:

- 判断 roofline 是否低于实测且量级合理。
- 找出小 batch decode GEMM、attention、MoE、collective 中误差大的 shape。
- 评估硬件峰值、dtype bytes、kernel_source、cudagraph/eager 数据是否对齐。

### 4.4 功能三: 模型级 real-vs-sim benchmark

目标: 将真实 vLLM benchmark 和 LLMInferSim virtual backend 放在同一批 case 上比较。

数据流:

```text
bench_cases.py
  -> cases.jsonl
  -> run_bench_suite.sh
  -> bench_compare.sh
       -> start real vLLM server
       -> run vllm bench serve
       -> start sim vLLM server
       -> run same vllm bench serve
       -> extract metrics
  -> analyze_bench.py
  -> suite-level gap report
```

suite:

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

输出:

```text
real_TTFT / sim_TTFT / gap
real_TPOT / sim_TPOT / gap
throughput gap
按 suite / TP / concurrency 聚合
```

用途:

- 校准模型级误差。
- 对比 batch、TP、long context、MoE 场景。
- 防止只对一个模型或一个 shape 过拟合。

### 4.5 功能四: 调试与归因

目标: 当模型级 gap 大时, 能逐层拆解到数据来源和算子形状。

典型数据流:

```text
模型级 benchmark gap
  -> StepCostTrace
  -> grouped operator entries
  -> 按 phase / op_kind / op_subtype / source 聚合
  -> 对关键 shape 查 OperatorDB 实测
  -> 对无 DB 的 op 查 roofline lower bound
  -> 输出误差归因
```

能回答的问题:

```text
TTFT gap 是 prefill attention 还是 GEMM?
TPOT gap 是 decode m 太小导致 launch/runtime 主导吗?
cudagraph 和 eager 数据是否混用了?
collector 是否缺当前 shape?
通信是否被跳过或低估?
vLLM worker/runtime overhead 是否超过 cost engine 估算?
```

### 4.6 功能五: 部署配置搜索准备

目标: 当前不实现完整搜索, 但核心仿真必须可被部署参数完整驱动。

数据流:

```text
candidate config
  -> DeployConfig
  -> ModelTemplate / Operator.parallel / Operator.runtime
  -> OperatorSignature
  -> CostRouter
  -> StepCostTrace metadata
  -> candidate metrics
```

后期搜索只应是外层循环:

```python
for candidate in search_space:
    deploy = candidate.to_deploy_config()
    result = simulator.run(workload, model_config, deploy)
    collect(result)
```

不在当前阶段提前实现:

```text
Pareto frontier
live validation
static envelope
自动 top-K 重跑
```

## 5. 核心抽象

### 5.1 GlobalStepWorkload / StepShape

`GlobalStepWorkload` 表示 framework-independent workload。

`StepShape` 是 cost engine 直接消费的 step 形状, 包含:

```text
phase: prefill / decode / mixed
prefill tokens
decode tokens
num sequences
context / kv length
block / cache metadata
runtime metadata
```

边界:

- `input_len / output_len / batch_size` 属 workload。
- `tp / pp / dp / ep / execution_mode / backend` 属 deploy/runtime。
- `w_byte / a_byte / kv_byte` 属 model/profile context。

### 5.2 DeployConfig

`DeployConfig` 是单次仿真的部署最小集:

```text
tp / pp / dp / ep_size
moe_tp / moe_ep_size
max_num_batched_tokens / max_num_seqs
block_size / num_gpu_blocks
execution_mode
backend / backend_version
```

它的目标是 search-ready, 不是把 workload 和 quantization 全塞进去。

后期配置搜索应只是外层循环:

```python
for candidate in candidates:
    result = simulator.run(workload, model_config, candidate.deploy_config)
```

### 5.3 Operator

`Operator` 是当前系统的 runtime op 主抽象。

每个具体 op 是一个语义类:

```text
GEMM
Attention
FusedMoE
Collective
ElementWise
Norm
Embedding
```

必须提供:

```python
shape -> dict
parallel -> dict
runtime -> dict
signature() -> OperatorSignature
roofline_spec() -> RooflineSpec
```

设计约束:

- `Operator` 不查询 DB。
- `Operator` 不决定走 DB 还是 roofline。
- `Operator.signature()` 只负责生成 DB key。
- `Operator.roofline_spec()` 只负责生成 roofline 输入。

### 5.4 GroupedStepPlan

生产 runtime 使用 `GroupedStepPlan`:

```text
GroupedStepPlan
  step_id
  phase
  groups: tuple[GroupedOperator]

GroupedOperator
  op: Operator
  count: int
  layer_indices: tuple[int, ...]
```

原因:

- Qwen / DeepSeek dense block 中大量 layer op shape 相同。
- per-layer 展开会让 Python 开销远大于仿真本身。
- grouped plan 能保持数学等价: `latency(group) = latency(rep_op) * count`。

`StepOpPlan` 如果存在, 只能作为测试或 debug DTO, 不应作为生产主路径。

### 5.5 StepCostTrace

cost 输出统一为:

```text
StepCostTrace
  step_id
  phase
  total_latency_s
  compute_time_s
  memory_time_s
  comm_time_s
  runtime_time_s
  entries: tuple[CostTraceEntry]
```

`CostTraceEntry` 记录:

```text
op_name / display_name
op_kind / op_subtype
source: roofline / operator_db / skipped / error
latency_s
roofline_s
metadata
```

grouped entry 应带:

```text
count
layer_indices
```

## 6. Operator Schema Contract

Operator schema 是 collector、runtime、OperatorDB 的共同契约。

硬规则:

```text
collector RawRecord.params
runtime Operator.shape/parallel/runtime
OperatorDB OperatorRecord.signature

必须 canonicalize 成同一个 OperatorSignature。
```

### 6.1 OperatorSignature

字段:

```text
op_kind
op_subtype
dtype
shape
parallel
runtime
```

signature 不包含:

```text
name
layer_idx
display_name
debug tags
```

### 6.2 GEMM

统一字段:

```text
op_kind = gemm
op_subtype = qkv_proj / o_proj / gate_up_proj / down_proj / lm_head / router / ...
dtype
shape = {m, n, k}
parallel = {tp}
runtime = {framework, framework_version, execution_mode, kernel_source}
```

`m` 是 token/batch 展开维度, 对 decode 性能非常敏感。

### 6.3 Attention

统一字段:

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
runtime = {attention_backend, kv_dtype, block_size, execution_mode, kernel_source}
```

### 6.4 MoE

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

`routing_distribution` 必须进入 key。

### 6.5 Collective

统一字段:

```text
op_kind = collective
op_subtype = allreduce / allgather / reduce_scatter / alltoall / p2p
dtype
shape = {message_bytes}
parallel = {world_size, tp, ep, node_count, gpus_per_node}
runtime = {backend=nccl, algo, protocol, topology, execution_mode}
```

## 7. ModelTemplate

模型模板直接构造 Operator, 不查 DB, 不算 latency。

### 7.1 Qwen Dense / MoE

Qwen dense layer grouped op pattern:

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

Qwen MoE layer pattern:

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

### 7.2 DeepSeek

当前 DeepSeek 支持边界应明确:

- V3 / MLA / MoE 可逐步支持。
- V4 sparse attention / hyper-connection 若 runtime 已删除, 不应在系统方案里作为当前能力承诺。
- 如果保留 V4 adapter, 应标为 config-only 或 migration reference。

### 7.3 Layer Partition

当不同 layer pattern 不一致时, 使用 `layer_partition` 将 layer 分桶:

```text
same pattern layers -> one GroupedOperator group
different pattern layers -> separate groups
```

## 8. Cost 层

### 8.1 CostRouter

当前只保留三种策略:

```text
roofline_only
operator_db_first
require_operator_db
```

不把 ModuleProfile 放在当前主路径优先级里。

### 8.2 OperatorDBBackend

职责:

```text
op.signature()
  -> store.lookup(signature)
  -> CostTraceEntry(source=operator_db, match_type=exact)
```

miss 时返回 `None`, 由 router 决定 fallback。

### 8.3 RooflineBackend

职责:

```text
op.roofline_spec()
  -> RooflineAnalyzer.analyze()
  -> CostTraceEntry(source=roofline)
```

RooflineSpec 包含:

```text
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

### 8.4 Communication

通信低层公式保留为 roofline primitive:

```text
core/cost/roofline/communication.py
```

`Collective` 可以作为 runtime op, 但 NCCL 公式库不必塞进每个 op class 内部。

### 8.5 ModuleProfile

ModuleProfile 不是当前主路径。

它可以后置用于:

- 整段 vLLM attention block profile。
- 框架层整段 module overhead。
- 当 operator-level + timeline correction 仍无法解释稳定偏差时再补。

## 9. Execution Mode 建模

### 9.1 eager / cudagraph 数据隔离

`execution_mode` 必须进入:

```text
Operator.runtime
OperatorSignature.runtime
OperatorDB key
StepCostTrace metadata
```

eager 和 cudagraph 不应互相 exact hit。

### 9.2 cudagraph 不是一个 op 属性

cudagraph 不应该简单建模成“每个 op 是否 graph mode”。

更合理的层次:

```text
step/runtime level:
  execution_mode
  capture size
  graph replay / piecewise graph overhead

operator level:
  shape / dtype / kernel_source / measured latency key
```

### 9.3 当前建议

首版:

- 让 OperatorDB 按 eager/cudagraph 分数据。
- roofline 主体不强行模拟每个 kernel launch。
- 模型级 TTFT/TPOT 稳定后, 再加 graph replay / pipeline correction。

## 10. Benchmark 与校准

### 10.1 Case-driven benchmark

benchmark 使用 case-driven suite:

```text
bench_cases.py      # 唯一 benchmark matrix
bench_compare.sh    # 执行器
run_bench_suite.sh  # suite 入口
analyze_bench.py    # 汇总
```

执行器不定义 batch / ISL / OSL / TP matrix。

### 10.2 Suite

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

### 10.3 默认约束

默认关闭:

```text
prefix_cache
chunked_prefill
```

默认不设置:

```text
max_num_seqs
```

suite 可以显式设置:

```text
max_model_len
max_num_batched_tokens
```

以避免模型 intrinsic max length 触发 vLLM scheduler 校验。

### 10.4 算子级报告

算子级 roofline vs real:

```bash
python scripts/report_operator_roofline_gap.py \
  --db-root collector/data/operator_db \
  --hardware RTX_4090 \
  --framework vllm \
  --framework-version 0.19.1 \
  --op-kind gemm
```

首轮重点:

- GEMM gap。
- 按 shape 展开。
- 按 eager/cudagraph 区分。
- attention/MoE/collective 先做 coverage, 再补公式级对比。

## 11. 配置搜索

配置搜索放后期, 不阻塞当前 roofline / OperatorDB 主链路。

要求当前 core search-ready:

```text
DeployConfig 完整参数化单次仿真
Operator.parallel/runtime 随 DeployConfig 改变
StepCostTrace metadata 可用于比较候选
```

搜索本身应是外层:

```text
candidate -> DeployConfig -> simulator.run() -> metrics
```

不要为了搜索提前在 core 里引入复杂 Pareto / live validation / static envelope。

## 12. 调试软件问题的价值

即使有真实芯片, LLMInferSim 仍有价值:

- 可拆解 model-level benchmark 与 operator-level 数据差距。
- 可快速定位 TTFT / TPOT 误差来自 prefill、decode、通信、runtime overhead 还是调度。
- 可在不跑长时间真实服务的情况下做部署参数对比。
- 可复现并解释 vLLM worker / scheduler 行为差异。
- 可作为组内统一性能归因语言。

## 13. 当前非目标

当前不作为主线:

```text
完整外部 scheduler simulator
ModuleProfile-first 路线
DeepSeek V4 runtime support
自动配置搜索
每个 eager kernel launch 的精细 pipeline 模拟
长期兼容 VirtualOp / LegacyDeployConfig
```

## 14. 后续收敛计划

与实施计划一致, 后续优先瘦身:

```text
1. GroupedStepPlan 成为唯一 runtime plan
2. 删除 RooflineOperator / KVTransfer legacy wrapper
3. 清 VirtualOp 命名和 schema 兼容层
4. 精简 CostRouter
5. benchmark 执行器 Python 化
6. scripts 目录分层
7. 明确模型支持边界
```

最终验收:

```bash
rg "core\\.cost_model|LegacyDeployConfig|VirtualOp|virtual_op|RooflineOperator|core\\.ops\\." llm_infer_sim tests scripts
conda run -n llm_sim pytest tests/core tests/adapters tests/scripts -q
```

允许例外:

```text
历史文档
migration notes
明确 deprecated wrapper 的提示
```
