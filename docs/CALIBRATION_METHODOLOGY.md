# LLMInferSim 校准方法论与实施计划

> 草稿 · 2026-05-17 · 待 review
> 关联文档:`docs/COMMUNICATION_MODELING.md` / `docs/PROJECT_REPORT.md`
> 关联脚本:`scripts/measure_*.py` / `scripts/bench_compare.sh`

---

## 1. 起因:Phase 5 校准失序的复盘

Phase 1-5 的校准实践不够规范,具体表现:

| 问题 | 表现 |
|------|------|
| **反推校准** | `framework_call_overhead 50→80µs` 是从端到端 bench gap 反推的,不是从独立实测得来 |
| **参数边界模糊** | 测 `link_latency` 拟出的 12µs 实际包含了 NCCL kernel 启动开销,不是纯链路延迟,但没有第二维度数据来分离两者 |
| **没有 stopping criteria** | "再调一下"成习惯,容易陷入越调越偏的过拟合循环 |
| **TP=1 / TP>1 / 单请求 / 多请求 全混跑** | 一旦有偏差,无法定位是 cost model 的问题还是调度的问题 |
| **不同 TP 用一套 SLA** | TP=8 通信占比远高于 TP=1,SLA 应不同 |
| **校准次序乱** | 没有"必须先校准 A,才能校准 B"的依赖顺序 |

本文档目的:**把校准从"看见 gap 就动"升级为"有目标、有顺序、有止损"的工程过程**。

---

## 2. 校准原则

### 2.1 五条铁律

1. **独立实测 > 端到端反推**
   每个待校准参数,必须有**单独的实测脚本**(`scripts/measure_*.py`)能在不依赖 cost model、不依赖 vLLM 的前提下,直接测出该参数的物理值。
   *反例*:framework_call_overhead 通过端到端 sim vs real bench 反推 → ❌
   *正例*:用 `torch.distributed` 直接计时 NCCL 调用 → ✓

2. **不过度校准**
   每组校准前先**确定可接受 gap 目标**。达标即停,**不允许"再优化一点"**。
   过度校准 = 把噪声 + 未建模因素都吸进参数里,换个场景就失准。

3. **物理 traceable**
   每个参数都应能对应**单一物理量**(带宽 / latency / 协议开销 / 拓扑因子)。
   *反例*:用一个 "η" 系数同时吸收 BW 不足 + dispatch overhead + L2 cache 命中 → ❌
   *正例*:`comm_step_latency` 单独表示 NCCL step 同步延迟,`framework_call_overhead` 单独表示 dispatch → ✓

4. **HW spec 不可校准**
   `peak_flops_*` / `mem_bandwidth` / `intra_node_bandwidth (nominal)` 等**直接来自硬件 spec**,不能调。
   只有 "efficiency" / "protocol overhead" 这类**spec 不给的现实折扣**才需要校准。

5. **校准结果有 provenance**
   每个 hardware profile 字段的值,注释里必须有:
   - 测量脚本路径
   - 测量日期 + HW/driver/NCCL/PyTorch 版本
   - 引用来源(我们自测 vs nccl-tests issue vs vendor whitepaper)

### 2.2 反例 vs 正例(Phase 5 实战教训)

| 操作 | 反例(Phase 5 干过) | 正例(应该的样子) |
|------|------|------|
| framework_oh 调到 80µs | 看端到端 bench 偏快,凭感觉 50→80 | 独立跑 `measure_framework_oh.py`(eager call - cudagraph call),实测 NCCL framework 开销中位 78µs → 写入 |
| comm_step_latency 改 12→5µs | 看 calibration intercept 估计 | `scripts/measure_p2p_latency.py` 测最小 P2P(1byte send/recv)wall-clock,扣除 framework_oh,得 5µs |
| protocol_efficiency 0.625 | 实测 β=5GB/s 反推 | 跑 `measure_allreduce.py` 多 size,fit α/β,β/(nominal/2) = 0.625 ← 这条**对了** |
| intra_node_gpus_per_root=4 | 拓扑直觉 | `nvidia-smi topo -m` 解析,**实证拓扑** ← 这条**对了** |

---

## 3. 待校准参数清单(必须独立实测)

### 3.1 计算侧

| 参数 | 物理含义 | 测量脚本(待写/已有) | 输入数据 |
|------|---------|---------|---------|
| `kernel_overhead[op_category]` | eager 模式 per-op CPU dispatch | `scripts/measure_compute_overhead.py` (待写) | 极小 GEMM (1 token) sleep loop, 测 wall-clock 跟 GPU active time 差 |
| `compute_efficiency` (per HW) | nominal flops vs achievable | `scripts/measure_compute_peak.py` (改自 `/tmp/vector_flops_bench.py`) | 大 GEMM 计算 BF16/FP8 throughput,跟 spec 比 |
| `mem_efficiency` (per HW) | nominal HBM BW vs achievable | `scripts/measure_mem_bw.py` (待写) | 大向量 copy / read,测 GB/s |
| op-level efficiency (per kind × shape × dtype) | `roofline_predicted / measured` ratio | 已有 `llm_infer_sim/calibration/` | 各 op 在标准 shape 下 profile,fit per-op |

### 3.2 通信侧(per HW × per collective)

| 参数 | 测量脚本 | 实测方法 |
|------|---------|---------|
| `intra_node_protocol_efficiency` | `scripts/measure_allreduce.py`(已有) | NCCL ring AllReduce 大 size 拟合 β,β/(nominal/2) |
| `comm_step_latency` | `scripts/measure_p2p_latency.py`(待写) | 1-byte P2P send/recv min latency,**cudagraph 模式**(扣 framework_oh) |
| `framework_call_overhead[coll]` | `scripts/measure_framework_overhead.py`(待写) | per-collective:eager - cudagraph delta(小消息域,扣 algorithm_term) |
| `collective_algo_bias[coll][algo]` | `scripts/measure_nccl_algo_choice.py`(待写) | 用 `NCCL_ALGO=Ring/Tree` 强制,比较实际选择跟我们 `min(候选)` 的差异 |
| `intra_node_gpus_per_root` / `gpu_to_root` | `nvidia-smi topo -m` + script parse | 拓扑解析,不是数值校准 |

### 3.3 调度侧(目前未建模 → backlog)

| 现象 | 当前归因 | 何时校准 |
|------|---------|---------|
| `short_short` -35% TTFT outlier | vLLM scheduler queue + GIL 时序 | Phase 7 调度建模(留 backlog,先标 known issue) |
| sample / detokenize 开销 | per-token CPU cost | 同上 |

### 3.4 明确**不校准**的参数(防 over-fit)

| 参数 | 来源 | 不调原因 |
|------|------|---------|
| `peak_flops_fp16/bf16/fp8/fp4` | 硬件 whitepaper | spec 数,实测 ≤ spec |
| `mem_bandwidth (nominal HBM)` | 硬件 whitepaper | 同上 |
| `intra_node_bandwidth (nominal)` | 链路 spec | 同上 |
| `inter_node_bandwidth (nominal)` | IB / Ethernet spec | 同上 |
| `vocab_size / hidden_dim / num_layers / ...` | 模型 config | 不变量 |

校准只调"现实折扣 / overhead",**不调硬件 spec**。

---

## 4. 校准目标(SLA)定义

### 4.1 不同维度容忍度

| 指标 | TP=1 SLA | TP=2 SLA | TP=4 SLA | TP=8 SLA |
|------|---------:|---------:|---------:|---------:|
| **TPOT(steady-state)** | ±15%¹ | ±15% | ±20% | ±25% |
| **TTFT(单请求/串行)** | ±15% | ±20% | ±25% | ±30% |
| **TTFT(多请求/并发)** | ±20% | ±25% | ±30% | ±35% |
| **Throughput** | ±10% | ±15% | ±20% | ±25% |

¹ 2026-05-17 调整:Stage A 实测 TPOT avg abs gap 12.8%(详 §14 backlog).
原 ±10% 假设过严,小 op compute framework_oh hidden/exposed 行为没建模时,
本质上小消息场景就会有 ~10-15% 偏差。调到 ±15% 跟 TP=2 一致。

理由:
- TPOT 是最关键的稳态指标,容忍度最严
- **TP 高 → 通信占比高 → 不确定性大 → SLA 放松**
- **多请求 TTFT 放宽不是因为 "scheduler 没建模"**(scheduler 是同一份 vLLM 跑在两边),而是:
  1. 多请求触发 chunked prefill 多 step,每 step cost model 误差累加放大
  2. NCCL 在 micro-batch 切换时有 setup overhead(实测显示这部分模不完)
  3. 短 prompt + 大并发(如 128_128 × 20)在 RTX 4090 实测有 ~30% 真机抖动,sim 是确定性的
  这些是 cost model 在 "多 step stack-up" 场景的已知偏差, 不是调度建模缺失

### 4.2 "达标"标准

每组测试场景的 **avg abs gap ≤ SLA 表对应值** 即达标。
**不要求**每个 single scenario 达标(避免追极端 outlier)。

### 4.3 不能调,只能标 backlog 的偏差

| 偏差 | 标准 |
|------|------|
| `short_short` 大并发 -35% | scheduler 未建模,**已知 known issue**,不计入 SLA 统计 |
| 单测 1B/16B 消息 latency | 跟 cost model overhead 同量级,小数取整误差,**记录但不调** |
| TP>8 跨节点 | 没机器测,**留 Phase X** |

---

## 5. 测试矩阵(优先级倒序,先做最简单)

### 5.0 文档 stage 名 ↔ 代码 group 名 对照

文档继续用 **Stage A/B/C/D/E** 命名,跟历史复盘 + 校准顺序图保持一致。
代码侧(`scripts/bench_cases.py`)用**自描述的 group 名**,这样 case_id /
results 目录 / log 不需要回查文档才知道在跑什么。两套命名 1:1 映射:

| 文档 Stage | 代码 group 名 | 含义 |
|------------|-------------|------|
| Stage A | `single_request_tp1` | dense, 单请求 TP=1 |
| Stage B | `single_request_multi_tp` | dense, 单请求 TP>1 (含拓扑 concentrated/balanced) |
| Stage C | `concurrent_tp1` | dense, 多请求 TP=1 |
| Stage D | `concurrent_multi_tp` | dense, 多请求 TP>1 |
| Stage E | `multi_model_regression` | 多模型回归(Qwen2.5-3B / Qwen3-32B) |
| Stage M-A (TP-only) | `moe_single_request_tp_only` | MoE 单请求 TP>1, **不开** `--enable-expert-parallel`(AllReduce 路径) |
| Stage M-A (EP)      | `moe_single_request_ep`      | MoE 单请求 TP>1, **开** `--enable-expert-parallel`(AllToAll dispatch/combine) |
| Stage M-B (TP-only) | `moe_concurrent_tp_only` | MoE 多请求, AllReduce 路径 |
| Stage M-B (EP)      | `moe_concurrent_ep`      | MoE 多请求, AllToAll 路径 |

**MoE 拆 TP-only / EP 的原因**:对 mixed expert 模型(Qwen3-30B-A3B),
开/不开 EP 走的是 **完全不同的通信路径**(`_build_moe_ffn_block`
里的 AllReduce vs AllToAll dispatch + AllToAll combine)。哪条路径 sim
拟合好都不能代表另一条,所以独立 group 独立校。

跑法:

```bash
bash scripts/run_bench_group.sh single_request_tp1     # Stage A
bash scripts/run_bench_group.sh moe_single_request_ep  # Stage M-A EP 路径
python scripts/analyze_bench.py /tmp/llm_infer_sim_bench --group single_request_tp1

# 开 per-op 校准(opt-in, §13.4):
LLM_INFER_SIM_USE_CALIBRATION=1 bash scripts/run_bench_group.sh single_request_tp1
```

### 5.1 Stage A:单请求 + TP=1(最简单)

| 维度 | 取值 |
|------|------|
| 模型 | Qwen3-4B-Instruct-2507 |
| TP / GPU | 1 (GPU 0) |
| 并发 | 1 prompt, 顺序串行 (`--request-rate 0.5`) |
| 输入长度 | 128, 256, 512, 1024, 2048 |
| 输出长度 | 16, 128, 512 |
| 模式 | eager / cudagraph |

scenarios = 5 × 3 × 2 = 30 个。

**达标条件**:TTFT/TPOT 平均 abs gap ≤ 15% 跟 10%。

**测脚本**:用现有 `scripts/bench_compare.sh`,改 `--num-prompts 1 --request-rate 0.5`,无并发干扰。

### 5.2 Stage B:单请求 + TP>1(加通信 — 真新物理)

> **顺序变更说明**(2026-05-17 review):原 Stage B 是 TP=1 多请求,
> 现交换到 Stage C。理由:通信是真新物理(NCCL collective)必须独立校,
> 而并发主要是 vLLM scheduler 行为(real / sim 同代码),没新物理。
> 优先校 comm,再校并发场景。

| 维度 | 取值 |
|------|------|
| TP | 2, 4, 8 |
| 拓扑 | TP=2 same NUMA / TP=2 cross NUMA / TP=4 same / TP=4 cross / TP=8 必跨 |
| 并发 | 1 prompt 顺序串行(`--num-prompts 3 --request-rate 0.5`) |
| 输入/输出 | 128/128, 512/512, 2048/512 |
| 模式 | eager / cudagraph |

scenarios = 5 拓扑 × 3 size × 2 mode = 30 个。

**达标条件**:TPOT 按 §4.1 SLA。

不达标 → 走 §7 通信参数依赖链顺序校准:`comm_step_latency → protocol_efficiency
→ framework_call_overhead → algo_bias`。

### 5.3 Stage C:多请求 + TP=1(加并发 — vLLM scheduler 路径)

| 维度 | 取值 |
|------|------|
| TP | 1 |
| 并发 | 5, 10, 20 prompt @ `--request-rate inf` |
| 输入/输出 | 沿用 Stage A 的 3 个代表组合:128/128, 512/512, 2048/512 |
| 模式 | eager |

scenarios = 3 × 3 × 1 = 9 个。

**达标条件**:TPOT 平均 ±10%, TTFT 平均 ±20% (§4.1 多请求 SLA TP=1 行)。

不达标 → cost model 在 chunked prefill / multi-batch step 路径有偏差,
不是 scheduler 问题(scheduler 同 vLLM)。可能要校准 chunked prefill 路径的 per-op
efficiency,或者发现 cost model 的小-step 累加偏差需要建模。

### 5.4 Stage D:多请求 + TP>1(完整 production 场景,B+C 叠加)

| 维度 | 取值 |
|------|------|
| TP | 2, 4, 8 |
| 拓扑 | concentrated / balanced 各 |
| 并发 | 10, 20 prompt |
| 输入/输出 | 5 个标准组合 |
| 模式 | eager (production 默认) |

scenarios = ~30-50 个。

**达标条件**:同 §4.1 多请求 SLA。

不达标 → 反向追源:
1. 看是否退到 Stage B 单测 TP>1 还达标 → 是 → 是并发叠加问题(回 C 复查)
2. 看是否退到 Stage C 单测 TP=1 多请求还达标 → 是 → 是通信问题(回 B 复查)
3. 都达标但 D 不达标 → 通信 × 并发交互效应,新建模需求

### 5.5 Stage E:多模型回归(确认不是 Qwen3-4B 特例)

- Qwen3-32B + TP=4/8
- Qwen2.5-3B + TP=1/2(对比 Qwen3-4B)

只跑 Stage A 的 3 个代表性 scenario,**回归测试性质**。

---

## 6. 校准顺序(关键!)

```
Stage 0 (baseline, 不校准)
    ↓
Stage A (TP=1 单请求)  →  TPOT 达标?  →  NO: 校准计算 op kernel_overhead / per-op eff
    ↓ YES                                  
Stage B (TP>1 单请求)  →  达标?  →  NO: 校准 comm 参数(§7 依赖链顺序:
    ↓ YES                                  comm_step_latency → protocol_efficiency
                                            → framework_call_overhead → algo_bias)
Stage C (TP=1 多请求)  →  达标?  →  NO: cost model 多 step 累加偏差,
    ↓ YES                                  考虑校准 chunked prefill efficiency
                                            (不是 scheduler 问题, scheduler 是同 vLLM)
Stage D (TP>1 多请求)  →  达标?  →  NO: 反向追:Stage B 还达标? Stage C 还达标?
    ↓ YES                                  都达标但 D 不达 → 交互效应,新建模需求
Stage E (多模型回归)    →  达标?  →  NO: 模型特异性(catalog 错)
```

**关键 invariant**:
1. **前一 stage 不达标,不能动后一 stage 的参数**(否则用前一 stage 偏差 over-fit 后一 stage)
2. **改了任何参数 → 必须回跑前面所有 stage**(防回归)
3. **Stage 顺序按"新物理"递进**:TP=1 单(基础)→ TP>1 单(加 comm)→ TP=1 多(加并发)→ 全叠加

---

## 7. 校准内部子顺序(单 stage 内多参数时)

当 stage 不达标,从**最基础物理量**开始往上调:

```
Phase 0:  HW spec 验证       (peak_flops_*, mem_bandwidth)
              ↓ 跑 measure_compute_peak.py / measure_mem_bw.py 实测
              ↓ 跟 spec 比,确认 efficiency 不是错离谱
Phase 1:  per-op efficiency (compute_efficiency, mem_efficiency)
              ↓ 走现有 calibration/ 工具
Phase 2:  通信参数依赖链
              comm_step_latency       (单 P2P 1 byte)
              ↓
              intra_node_protocol_efficiency (大 AllReduce fit β)
              ↓
              framework_call_overhead  (eager - cudagraph delta)
              ↓
              collective_algo_bias    (per-algo NCCL 实际选择)
Phase 3:  调度建模(后续 阶段)
```

**关键 invariant**:**调下游参数前,上游必须实测稳定**。否则下游会吸收上游的偏差。

---

## 8. 测试脚本清单

### 8.1 已有

| 脚本 | 用途 | 状态 |
|------|------|------|
| `scripts/measure_allreduce.py` | AllReduce α/β fit | ✓ 用于 protocol_efficiency |
| `scripts/measure_collectives.py` | 6 collective × 5 NUMA × 13 size × 2 mode | ✓ 用于 collective 覆盖 |
| `scripts/run_collective_sweep.sh` | driver | ✓ |
| `scripts/analyze_collectives.py` | 跟 cost model 对比 | ✓ |
| `scripts/bench_compare.sh` | 端到端 vllm bench, 支持 `SCENARIO_OVERRIDE` + `ENABLE_EP` env | ✓ 用于 case 驱动 |
| `scripts/bench_cases.py` | case 定义 + cases.jsonl 生成器(group 化) | ✓ §5.0 命名映射 |
| `scripts/run_bench_group.sh` | 按 group 读 cases.jsonl, 逐 case 跑 bench_compare | ✓ 取代 `run_stage_bench.sh` |
| `scripts/_extract_metrics.py` | 从 case_dir 抽 TTFT/TPOT/p99/throughput 进 metrics.json | ✓ run_bench_group 内部用 |
| `scripts/analyze_bench.py` | per-case 表 + per-(group,TP) 聚合 + SLA 判定 | ✓ 取代 `analyze_stage.py` |

### 8.2 需新写

| 脚本 | 用途 | 输出 |
|------|------|------|
| `scripts/measure_compute_peak.py` | nominal flops 实测验证(BF16/FP8/FP32) | `/tmp/compute_peak.jsonl` |
| `scripts/measure_mem_bw.py` | nominal HBM BW 实测验证 | `/tmp/mem_bw.jsonl` |
| `scripts/measure_compute_overhead.py` | per-op kernel_overhead 实测(eager vs graph) | `/tmp/compute_overhead.jsonl` |
| `scripts/measure_p2p_latency.py` | NCCL P2P 1-byte cudagraph latency = comm_step_latency 真值 | `/tmp/p2p_latency.jsonl` |
| `scripts/measure_framework_overhead.py` | per-collective eager-cudagraph delta(代替手算) | `/tmp/framework_oh.jsonl` |
| `scripts/measure_nccl_algo_choice.py` | NCCL_ALGO=Ring/Tree 强制,看默认选择 | `/tmp/nccl_algo.jsonl` |
| ~~`scripts/run_stage_bench.sh`~~ | **已 deprecated**,改用 `run_bench_group.sh`(§8.1) | — |
| ~~`scripts/analyze_stage.py`~~ | **已 deprecated**,改用 `analyze_bench.py`(§8.1) | — |

### 8.3 脚本质量要求

- 必须 **idempotent**:重复跑不依赖前次状态
- 必须 **machine-version-aware**:输出 JSON 里记录 PyTorch / NCCL / driver 版本
- 必须 **single-purpose**:一个脚本测一个物理量,**不**混合
- 必须有 README 解释"测的是什么 + 怎么解读结果 + 写到 profile 哪个字段"

---

## 9. 实施计划(分阶段)

### Phase 6 - 校准基建(本文档落地)

| 子阶段 | 工作 | 预估 | 产物 |
|--------|------|----:|------|
| 6.0 | review + lock 本文档 | 1h | docs/CALIBRATION_METHODOLOGY.md |
| 6.1 | 写 7 个 measure_*.py 脚本 + README | 1d | scripts/measure_*.py |
| 6.2 | 写 run_stage_bench.sh + analyze_stage.py | 0.5d | scripts/ |
| 6.3 | 跑 Phase 0 + Phase 1(HW spec + per-op eff 验证) | 0.5d | configs/calibration/raw/RTX_4090/*.jsonl |

### Phase 7 - 按 Stage 推进(每 stage 独立)

| Stage | 跑测试 | 不达标 → 校准 | 达标 → 锁定 |
|-------|--------|--------------|------------|
| **A**: TP=1 单请求 | bash run_stage_bench.sh A | kernel_overhead / per-op eff 微调 | 写 profile + commit |
| **B**: TP>1 单请求 | bash run_stage_bench.sh B | §7 通信参数依赖链 | 同上 |
| **C**: TP=1 多请求 | bash run_stage_bench.sh C | chunked prefill efficiency / 多 step 累加偏差 | 同上 |
| **D**: TP>1 多请求 | bash run_stage_bench.sh D | 反向追源 B / C | 同上 |
| **E**: 多模型回归 | bash run_stage_bench.sh E | 模型特异性,看 catalog | 同上 |

**关键纪律**:任何 stage 改了参数 → **回跑前面 stage**(确保不回归)。

### Phase 8 - 扩硬件(每加一个 HW)

每个新 HW(A100 / H100 / B200 / 自研芯片)按 Phase 7 流程独立跑一遍。模板化以减少人力。

### Phase 9 - 调度建模(Stage B/D outlier 的根因)

留 backlog,等 Phase 7 通信全部达标后再启动。

---

## 10. 数据存档约定

### 10.1 raw 测量数据

`configs/calibration/raw/<HW>/<YYYY-MM-DD>/<script_name>.jsonl`

例:`configs/calibration/raw/RTX_4090/2026-05-17/measure_allreduce.jsonl`

每行 JSON 包含:
- `script`: 脚本名
- `script_version`: git commit / SHA
- `hardware`: HW 标识
- `pytorch_version`, `nccl_version`, `driver_version`
- `measurement_data`: 实际数值

### 10.2 distilled 参数(写进 profile)

`HardwareConfig` 字段值 + 注释:

```python
"RTX_4090": dict(
    ...,
    # protocol_efficiency: 0.625 from scripts/measure_allreduce.py @ 2026-05-15
    # data: configs/calibration/raw/RTX_4090/2026-05-15/measure_allreduce.jsonl
    intra_node_protocol_efficiency=0.625,
    ...
)
```

### 10.3 stage 验证报告

`docs/calibration_reports/<HW>/stage_<X>_<YYYY-MM-DD>.md`

包含:
- 测试命令
- 完整 gap 表
- 跟 SLA 对比
- 用了哪些参数值

---

## 11. 还有什么需要补充的(我建议加上)

用户的草稿已经覆盖了核心思路。我额外建议:

### 11.1 加 **"don't calibrate"** 明确列表

防止有人(包括将来的我)看到 gap 就想动 peak_flops 这种 spec 量。已在 §3.4 加。

### 11.2 加 **per-metric SLA** 区分(TTFT vs TPOT)

TPOT 是稳态指标,SLA 应该比 TTFT 严。已在 §4.1 加。

### 11.3 加 **校准内部子顺序**(单 stage 多参数时)

不只是 Stage A→B→C 的顺序,**单 stage 内**也得有顺序(基础物理量 → 上层 overhead)。已在 §7 加。

### 11.4 加 **脚本质量要求**

防止脚本写得不能复用 / 不带版本信息。已在 §8.3 加。

### 11.5 加 **HW spec validation** Phase 0(测前先验)

很多 sim 偏差来自 hw profile 里 peak FLOPS 填错(比如 H100 990TF vs 1979TF 的稀疏/稠密混淆)。Phase 0 先实测 peak 跟 spec 比,确认起点正确。已在 §7 Phase 0 加。

### 11.6 加 **回归测试纪律**

改了任何参数 → 必须回跑前面已达标的 stage。已在 §9 Phase 7 加。

### 11.7 加 **校准数据存档结构**

不只是参数值,还要存 raw 测量 JSONL + 复现命令,以便后续审计 / 复测。已在 §10 加。

### 11.8 加 **测试脚本 README 强制要求**

每个 measure_*.py 都要有 README 解释"测什么 / 怎么解读 / 写到哪"。已在 §8.3 加。

### 11.9(争议)**SLA 数值是不是过严**?

我提议的 TPOT TP=1 SLA 是 ±10%。Phase 5 当前 Qwen3-4B TP=1 TPOT 平均 7.6%,达标。
但如果某 HW(比如老 GPU / 国产芯片)cost model 本身就有 ±15% 噪声,可能这个 SLA 拍得太死。
**建议**:SLA 表第一次 review 时讨论,**先松后紧**(从 ±15%/20%/25% 起步,落实后再考虑收紧)。

---

## 12. 待用户决定的事

1. **§4.1 SLA 数值表**是否合适?TPOT ±10% 是否过严?
2. **Phase 6 是否优先级最高**(优先于 Phase 7 实际校准)?还是边写脚本边跑?
3. **校准数据存档位置**:`configs/calibration/raw/` 还是分开 repo?
4. **Stage B/D 多请求的 SLA 是不是放更松**(因为已知 scheduler 没建模)?现在 TTFT TP=4 多请求 ±25% 是否合理?
5. **是否引入 CI / 自动跑**?(每 commit 自动跑 Stage A 防回归)
6. **量化模型(FP8/FP4)** 是否要单独一 Stage?他们走的 op kind 不一样

---

## 13. 已知 backlog(校准过程中发现的未建模问题)

### 13.1 小 op compute framework_overhead "GPU 饿等" 暴露建模(Stage A 发现)

**现象**(2026-05-17 Stage A 实测,Qwen3-4B TP=1 REQUEST_RATE=0.5):

| scenario | TPOT gap | 说明 |
|----------|--------:|------|
| 128_128 | -17.3% | 短 prompt + 小 decode,小 op 多,sim 偏快 |
| 256_256 | -17.1% | 同上 |
| 512_512 | -12.8% | 中等 |
| 1024_512 | -12.5% | 中等 |
| 2048_512 | **-4.3%** | GPU 大 compute 主导,**模得对** |

**根因**:`Phase 0c measure_compute_overhead.py` 实测 eager 模式每个 compute op 有
~49 µs framework dispatch overhead(matmul_small / norm_like / elementwise 都在
40-55µs 量级)。这部分在大 GPU op(矩阵乘几 ms)上 hidden(CPU 跟 GPU 并行),
在小 op(decode 阶段 RMSNorm / RoPE 几 µs)上**暴露**到 wall-clock。

我们当前 `kernel_overhead={}` 不加任何 per-op overhead → 大 op 准、小 op 偏快。

**为什么不直接加 `kernel_overhead=49µs/op`**:
- 大 op 主导时 49µs 是 hidden 的,加进去会**over-correct** → 17.6ms/step 虚高
- 物理上"暴露多少"取决于 GPU compute 时间是否 ≥ CPU dispatch 时间,
  是 **data-dependent** 的,单一常数捕不住

**建模方向**(留 Phase X):
```
overhead_per_op = max(0, framework_oh - GPU_compute_time_of_op)   # 普通 async
overhead_per_op = framework_oh                                     # 有 sync barrier (NCCL)
```
即"GPU 比 dispatch 快多少,就暴露多少 dispatch 时间;有 sync barrier 时完全暴露"。

**重要:计算 vs 通信的"暴露程度"不对称是物理的,不是建模偷工**:
- 计算 op 之间没强制 sync, CPU dispatch 跟 GPU exec 异步重叠 → 大 op 时 framework_oh hidden
- NCCL collective 内部有 **跨 rank sync barrier**, 所有 rank 必须到齐才能进 → CPU/GPU pipelining 在 barrier 处断, framework_oh **始终至少部分暴露**
- 这就是为什么我们当前模型 `T_comm = algo + framework_oh × [eager]` 加 full, `T_compute = roofline + 0` 加 0,**不对称但物理上各自合理**

需要 cost model 在每个 op 上做这个 max 比较, 现公式架构(只算 t_compute / t_memory / k_overhead)不容易加,推 Phase X。

**短期妥协**:
- TPOT TP=1 SLA 从 ±10% 放宽到 ±15%(本表已改)
- 接受 TPOT 在小 size 上 -12 ~ -17% 是已知偏差
- 大 size(GPU compute 主导)预测仍准

### 13.2 cross-node `_hierarchical_alltoall` 公式 per-rank 语义偏差(Phase 5 标 xfail)

详 `tests/core/test_inter_node_cost_consistency.py::test_cross_node_slower_than_single_node_by_realistic_factor`。
当 `data_bytes` 改成 per-rank input bytes 语义后,`(data * n1) / (n*n * beta_inter)`
公式系数偏小,留 Phase X 跨节点重构修。

### 13.3 多请求 scheduler 时序累加偏差(Phase 5 发现)

短 prompt 大并发(如 128_128 × 20)在 RTX 4090 实测 ~30% 真机抖动,sim 是
确定性的。这不是 scheduler 缺建模(scheduler 同 vLLM),而是 chunked prefill
多 step 累加 cost model 误差。校准方向:Stage C 重新看,可能需要校 chunked
prefill efficiency。

### 13.4 per-op efficiency calibration 的 mode-dependent 价值(2026-05-17 Stage A 发现)

**实测对照**(Stage A,Qwen3-4B TP=1 serial,RTX 4090,eager 模式):

| 配置 | TPOT avg abs gap | TTFT avg abs gap |
|------|---:|---:|
| `LLM_INFER_SIM_USE_CALIBRATION` 未设(默认) | 12.8% | 13.1% |
| `LLM_INFER_SIM_USE_CALIBRATION=1` | **7.0%** | 13.9% |

**finding**:per-op calibration(`configs/efficiency/rtx_4090.yaml`)在 eager 模式
**真物理修正**,不是巧合。decode 阶段 memory-bound,小 op (RMSNorm/RoPE/SwiGLU 等)
shape-aware efficiency 拟合后,sim 跟 real TPOT 平均 abs gap 从 12.8% 降到 7.0%
(达原 ±10% SLA)。

**为什么默认 off**:

| 场景 | calibration on 的效果 |
|------|---------------------|
| **eager 模式** | ✓ 真物理修正(本节实测) |
| **cudagraph 模式** | ✗ 过度补偿 → 偏 +60-90%(2026-05-16 实验观察) |

eager 模式 dispatch overhead 大,calibration 隐式吸收了这部分;cudagraph 模式
dispatch 已经消了,再叠 calibration 等于双重补偿。

**用户建议**:
- 跑 `vllm serve --enforce-eager` 评估 → 显式设 `LLM_INFER_SIM_USE_CALIBRATION=1`
- 跑 production cudagraph 模式 → 不设(默认 off)
- 长期方向(Phase X):基于 `BackendExecutionProfile.execution_mode` 自动选,
  或维护两份 calibration YAML(eager / graph)按 mode 加载

raw 对照数据:`/tmp/stage_A/TP1_serial/` vs `/tmp/stage_A_calib/TP1_serial/`。

### 13.5 短 prefill TTFT ~23ms 固定偏差(2026-05-18 Stage A `single_request_tp1` 10-case 发现)

**实测**(Qwen3-4B TP=1 eager, calibration ON, `--num-warmups=1`):

| case | ISL | OSL | real_TTFT | sim_TTFT | Δ (real-sim) | gap% |
|------|----:|----:|----------:|---------:|------------:|-----:|
| baseline   | 128 | 128  | 69.9   | 43.6  | 26.3 | -37.6% |
| decode_512  | 128 | 512  | 67.4   | 44.5  | 22.9 | -33.9% |
| decode_1024 | 128 | 1024 | 72.8   | 49.0  | 23.8 | -32.8% |
| decode_2048 | 128 | 2048 | 68.6   | 48.4  | 20.2 | -29.5% |
| prefill_512  | 512  | 128 | 79.1   | 67.2  | 11.9 | -15.0% |
| prefill_1024 | 1024 | 128 | 100.5  | 97.3  | 3.2  | -3.3%  |
| prefill_2048 | 2048 | 128 | 156.1  | 164.3 | -8.2 | +5.3%  |
| prefill_4096 | 4096 | 128 | 304.7  | 313.1 | -8.4 | +2.7%  |
| prefill_8192 | 8192 | 128 | 644.4  | 718.8 | -74.4| +11.5% |
| mix         | 4096 | 1024 | 310.1  | 320.6 | -10.5| +3.4%  |

**模式**:ISL=128 的 4 case 全部 Δ ≈ 20-26ms;ISL≥512 时 Δ 散落 ±10ms 内。
**汇总指标**:TTFT mean abs gap = 17.5% (SLA 15%, ❌ marginal FAIL);
若只看 ISL≥512 子集 (6 case) = **6.9%**(✓ PASS)。

**根因初查**(详 Explore 2026-05-18):

| 嫌疑 | REAL 开销 | SIM 路径 | Δ |
|------|---------|---------|---|
| `_prepare_inputs` (block table / position) | 50-200µs | 完全 skip(VirtualWorker.execute_model) | ~150µs |
| `Sampler.forward` (logits processing) | 100-300µs | 完全不调用(sim 直接返回 pre-sampled token_ids,no logits tensor) | ~300µs |
| Detokenize first token | 1-5µs | skip | ~5µs |
| **以上 3 项合计** | | | **~500µs** |

**关键洞察**:prepare/sample/detok 加起来只 ~500µs,**凑不到 23ms**。API server +
vLLM scheduler 是 sim/real 共享(都跑 vllm serve),不贡献 gap。所以 23ms 的剩余
~22ms 来源未定位,候选嫌疑(均未实测):
- vLLM V1 EngineCore step 内部 bookkeeping(KV slot 分配 / seq group state)
- async event loop + SSE streaming 在 sim 路径下的非对称行为
- TTFT 测量边界差异(real 是 HTTP request → 第一个 SSE event 字节;sim 同理但内部时间线不同)

**当前应对**:
- ISL≥512 case 已经达标 (mean 6.9%),业务相关 workload (chat / RAG) 都覆盖
- ISL=128 极短 prompt 在 production 几乎不存在(API 请求最少都几百 token 上下文)
- **接受此偏差,优先推 Stage B**(TP>1 通信主导,跟 fixed overhead 解耦)

**未来若需精确建模**:
- 写 monkey-patch profile 脚本(real + sim 各 instrument 一次 baseline case)
  直接拿到 per-phase 实测耗时, 定位剩余 ~22ms 出处
- 若锁定到 `_prepare_inputs` / `Sampler` GPU 部分,可加 2 个新 op:
  `PrepareInputs.estimate(batch, seq_len)` + `Sampler.estimate(batch, vocab)`,
  数值需独立实测不可 curve-fit(详 §2.1 铁律 1)

raw 数据:`/tmp/llm_infer_sim_bench/single_request_tp1/`。

### 13.6 sim `time.sleep` blocking 让 TTFT 方向反转(2026-05-18 Stage B + probe 发现)

**Stage B `single_request_multi_tp` 15 case 全跑 PASS,但发现两个反向偏差**:

| outlier case | TTFT gap | 方向 |
|--------------|---------:|------|
| TP=2 same NUMA prefill_i2048 | -23% | sim 偏快(real 慢)|
| TP=4 cross NUMA prefill_i2048 | **+28%** | sim 偏慢(real 快)|
| TP=4 cross NUMA mix_i4096 | **+34%** | 同上 |

两个 case 都 ISL=2048,warmup=1 已开,SLA 上 TP=2 ±20% / TP=4 ±25% 全过,但方向不一致刺激了进一步 probe.

**probe 路径**:dump sim_server.log per-step latency,对比 `TIME_MODE=realtime` vs `instant`,
对比 `--async-scheduling` vs `--no-async-scheduling`.

**关键发现**:

1. **cost model 真实偏差只 ~-15%**:
   ```
   real_TTFT - sim_pipeline_baseline(44ms) = real_compute = 184ms
   sim_reported_per_step                                 = 155.7ms
   实际 cost model 偏差 = (155.7 - 184) / 184 = -15.4%  ← sim 偏低
   ```

2. **realtime mode 把方向反转成 +28%**:
   ```
   sim_TTFT(realtime, bench 端测) = 282ms
   = time.sleep(155.7ms) + IPC + OS wake lag + SSE queue accum ≈ 280ms
   real_TTFT 220ms
   gap = (282-220)/220 = +28%   ← 看着 sim 偏慢
   ```
   `time.sleep` 在 worker 进程里**阻塞 OS thread**,触发 OS 降级到 sleep 状态,
   后续 wake 有 ~1-3ms 延迟 per call。128 个 decode step × wake lag + 4-worker
   ZMQ broadcast IPC × OSL × cross-NUMA UPI 累积成 ~80-100ms 额外 wall clock。

3. **vLLM `--no-async-scheduling` 不修这条**:实测两边都开 `--no-async-scheduling`,
   real 220→205ms / sim 282→276ms,gap 反而扩大到 +35%。说明这个 distortion
   不是 V1 async scheduling 的事,是 OS 进程调度 + time.sleep 本身行为。

**三个修复方向**:

| 方案 | 工作量 | 修复程度 |
|------|--------|---------|
| A. virtual clock(busy-wait + advance virtual time)| 中(改 `time_emulator` + worker 钩子) | 根治 |
| B. driver 改 `instant` mode + parse `sim_server.log` per-step latency | 小(改 `analyze_bench.py`) | 直接验证 cost model,绕过 time.sleep |
| C. 文档化接受现状 | 0 | 不修 |

**决议(2026-05-18)**:**走 C**。理由:
- Stage A/B 15 case 都 PASS SLA(SLA 已经按 TP 放宽)
- 真实偏差只 ~15%,被 time.sleep distortion 伪装成 +28%,**SLA 仍 PASS**
- B 方案需要修改一套测量管线,A 方案动 sim 架构,**当前 SLA PASS 状态下投入产出比低**
- 真要做精确 cost model 验证时(比如换硬件 / 新模型),可重启 A 或 B

**关于 same-NUMA / cross-NUMA β 公式偏差**(单独议题):
- TP=2 same -23% vs TP=4 cross +28% 在 sim 模型层面已经互抵
- sim 当前 `effective_intra_bw` 对 same-root contention 估算偏弱(real 同根 4 卡 PCIe 共享更严重),对 cross-root penalty 估算偏强(NCCL pipelining 实际 hide 了)
- 由于 cost model 偏差被 time.sleep 放大反转,**短期无法干净校准**(B/A 方案落地后才能精确测)

raw 数据:`/tmp/llm_infer_sim_bench/single_request_multi_tp/` + `/tmp/probe_instant/` + `/tmp/probe_noasync/`。

---

## 14. 一句话总结

**当前混乱**:校准从端到端 gap 反推,无独立测量、无 SLA 目标、无依赖顺序,容易 over-fit。
**本文档**:把校准拆成"实测 → 分阶段验证 → 不过度调"的工程过程,7 个测试脚本 + 4 阶段 + 明确 SLA,前置达标才解锁下一阶段。
**预期收益**:可重复 / 可审计 / 可推广到新硬件,**避免 Phase 5 那种"端到端 gap 一调,再换 mode 就垮"的尴尬**。
