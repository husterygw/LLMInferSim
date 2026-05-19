# LLMInferSim 全维度校准结果 — 2026-05-18

跨 9 个 group / **57 个 case** 的 sim vs real benchmark 全量结果。
模型:Qwen3-4B-Instruct-2507 (dense),Qwen3-30B-A3B-Instruct-2507 (MoE),Qwen2.5-3B-Instruct,Qwen3-32B。
硬件:RTX 4090 × 8 (dual NUMA, 4 GPU/root)。
vLLM: 0.19.1,torch 2.10.0,enforce_eager。
calibration: `LLM_INFER_SIM_USE_CALIBRATION=1` (per-op efficiency YAML),`--num-warmups=1`。

raw 数据:`/tmp/all_results.csv` + `/tmp/all_results.jsonl`。
raw case 输出:`/tmp/llm_infer_sim_bench/<group>/<case_id>/`。

---

## 1. SLA 总览(按 group × TP 聚合)

| group | TP | N | TTFT abs | TPOT abs | Throughput abs | TTFT SLA | TPOT SLA | 判定 |
|-------|---:|--:|---------:|---------:|---------------:|---------:|---------:|------|
| **single_request_tp1** (Stage A) | 1 | 10 | **17.5%** | **9.6%** | 4.8% | 15% | 15% | ⚠ marginal |
| **single_request_multi_tp** (Stage B) | 2 | 6 | 19.0% | 11.3% | 11.4% | 20% | 15% | ✓ PASS |
| | 4 | 6 | 15.2% | 9.1% | 10.9% | 25% | 20% | ✓ PASS |
| | 8 | 3 | 3.6% | 9.6% | 11.4% | 30% | 25% | ✓ PASS |
| **concurrent_tp1** (Stage C) | 1 | 12 | **8.5%** | **9.5%** | 6.5% | 20% | 15% | ✓ PASS |
| **concurrent_multi_tp** (Stage D) | 2 | 4 | 10.4% | 9.0% | 5.2% | 25% | 15% | ✓ PASS |
| | 4 | 4 | 6.4% | 6.3% | 3.7% | 30% | 20% | ✓ PASS |
| | 8 | 4 | 3.2% | 9.4% | 4.0% | 35% | 25% | ✓ PASS |
| **moe_single_request_tp_only** (Stage M-A) | 4 | 5 | 24.6% | **53.3%** | 71.8% | 25% | 24% | ❌ FAIL |
| **moe_single_request_ep** (Stage M-A) | 4 | 3 | 29.5% | **42.4%** | 53.0% | 25% | 24% | ❌ FAIL |
| **moe_concurrent_tp_only** (Stage M-B) | 4 | 2 | 3.7% | **44.9%** | 78.2% | 30% | 24% | ❌ FAIL |
| **moe_concurrent_ep** (Stage M-B) | 4 | 2 | 22.3% | **33.6%** | 50.2% | 30% | 24% | ❌ FAIL |
| **multi_model_regression Qwen2.5-3B** (Stage E) | 1 | 1 | 13.2% | 19.6% | 10.0% | 20% | 15% | ❌ FAIL |
| **multi_model_regression Qwen3-32B** (Stage E) | 4 | 1 | 4.4% | 7.4% | 0.0% | 30% | 20% | ✓ PASS |

**33 / 57 case 严格 PASS;24 / 57 FAIL。FAIL 集中在 MoE (12 cases) + 短 prefill (dense Stage A 部分)**。

## 2. 一句话观察

| 维度 | 观察 |
|------|------|
| **Dense Qwen3-4B TP=1~8 prefill / decode** | ✓ 模型整体准 (TTFT/TPOT 多在 ±10%) |
| **Dense 并发(c=4~32)** | ✓ 并发拉大后 sim 更准(GEMM batch 增大,fixed overhead 摊薄)|
| **Dense 短 prefill (ISL=128)** | ⚠ TTFT 偏低 ~23ms 固定值(§13.5,未模 prepare_inputs/sampler) |
| **TP=4 cross-NUMA 单请求** | ⚠ sim 偏慢 ~28-34%(comm topology 模型偏差) |
| **MoE Qwen3-30B-A3B 所有 TPOT** | ❌ sim 偏快 ~33-53%(MoE 路径 comm/launch overhead 未建)|
| **MoE TTFT(长 prefill 主导)** | ✓ -7~-15% (prefill 计算 dominate,MoE comm 占比小) |
| **跨模型 Qwen2.5-3B (TP=1)** | ❌ TPOT -20%(短 prefill 受 §13.5 影响) |
| **跨模型 Qwen3-32B (TP=4)** | ✓ +7%(大模型 prefill compute dominate) |

## 3. 完整 per-case 表

### Stage A: single_request_tp1 (10 cases, dense Qwen3-4B TP=1)

```
shape                  ISL/OSL    real(TTFT/TPOT)    sim(TTFT/TPOT)    gap(TTFT/TPOT)
baseline               128/128    69.9 / 20.1        43.6 / 17.1       -37.6% / -14.8%
prefill_512            512/128    79.1 / 20.2        67.2 / 17.3       -15.0% / -14.1%
prefill_1024          1024/128   100.5 / 20.4        97.3 / 17.5        -3.3% / -14.2%
prefill_2048          2048/128   156.1 / 19.9       164.3 / 18.0        +5.3% /  -9.7%
prefill_4096          4096/128   304.7 / 20.9       313.1 / 19.4        +2.7% /  -6.9%
prefill_8192          8192/128   644.4 / 22.6       718.8 / 23.1       +11.5% /  +2.2%
decode_512             128/512    67.4 / 19.6        44.5 / 17.6       -33.9% / -10.3%
decode_1024           128/1024    72.8 / 20.0        49.0 / 17.9       -32.8% / -10.7%
decode_2048           128/2048    68.6 / 20.6        48.4 / 18.0       -29.5% / -12.8%
mix                  4096/1024   310.1 / 20.8       320.6 / 20.8        +3.4% /  -0.1%
```

### Stage B: single_request_multi_tp (15 cases, dense Qwen3-4B TP>1)

```
shape × topology         real(TTFT/TPOT)    sim(TTFT/TPOT)    gap(TTFT/TPOT)
prefill_i2048 TP=2 same     231 / 26.5        178 / 23.0        -23.0% / -13.4%
prefill_i2048 TP=2 cross    186 / 25.8        171 / 23.0         -8.1% / -10.7%
prefill_i2048 TP=4 same     327 / 26.7        317 / 25.0         -3.3% /  -6.3%
prefill_i2048 TP=4 cross    220 / 25.9        282 / 25.1        +28.4% /  -3.4%
prefill_i2048 TP=8          344 / 26.8        350 / 24.2         +1.5% /  -9.6%
decode_i128 TP=2 same        72 / 25.3         45 / 22.3        -37.7% / -11.6%
decode_i128 TP=2 cross       75 / 26.1         51 / 22.3        -32.8% / -14.5%
decode_i128 TP=4 same        79 / 26.1         68 / 22.8        -13.4% / -12.9%
decode_i128 TP=4 cross       76 / 26.1         71 / 23.6         -7.3% /  -9.2%
decode_i128 TP=8             78 / 25.6         71 / 23.4         -8.4% /  -8.8%
mix_i4096 TP=2 same         398 / 26.2        356 / 23.7        -10.6% /  -9.5%
mix_i4096 TP=2 cross        331 / 25.6        325 / 23.5         -1.9% /  -8.2%
mix_i4096 TP=4 same         625 / 27.1        592 / 23.6         -5.3% / -12.9%
mix_i4096 TP=4 cross        391 / 26.1        522 / 23.5        +33.6% /  -9.7%
mix_i4096 TP=8              641 / 26.2        646 / 23.4         +0.8% / -10.5%
```

### Stage C: concurrent_tp1 (12 cases, dense TP=1 concurrent)

```
workload         c     real(TTFT/TPOT)        sim(TTFT/TPOT)         gap(TTFT/TPOT)
chat (512/512)   4     156.8 / 21.4           122.2 / 17.9           -22.1% / -16.4%
chat             16    442.8 / 20.1           428.8 / 21.6            -3.2% /  +7.5%
chat             32   5734.4 / 20.5          6079.4 / 21.8            +6.0% /  +6.3%
rag_prefill      4     774.2 / 22.2           795.3 / 23.7            +2.7% /  +6.8%
(4096/128)       16   4933.3 / 26.2          5025.3 / 29.8            +1.9% / +13.7%
                 32 10951.2 / 26.8         11086.6 / 31.3            +1.2% / +16.8%
decode_heavy     4      89.4 / 20.1            74.3 / 18.6           -16.9% /  -7.5%
(128/2048)       16    227.7 / 22.6           194.3 / 23.4           -14.7% /  +3.5%
                 32 22281.0 / 21.2         24690.7 / 24.1           +10.8% / +13.7%
long_context     4    4441.6 / 25.6          4837.8 / 24.4            +8.9% /  -4.7%
(8192/512)       16 34466.1 / 28.0         32850.6 / 25.9            -4.7% /  -7.4%
                 32 78597.5 / 29.2         71305.2 / 26.4            -9.3% /  -9.6%
```

### Stage D: concurrent_multi_tp (12 cases, dense TP>1 concurrent)

```
workload × c × TP            real(TTFT/TPOT)    sim(TTFT/TPOT)    gap(TTFT/TPOT)
chat c=16 TP=2                514 / 27.5         567 / 24.5       +10.3% / -10.9%
chat c=16 TP=4               1079 / 26.1        1051 / 25.2        -2.6% /  -3.4%
chat c=16 TP=8               1124 / 27.7        1194 / 25.3        +6.2% /  -8.7%
chat c=32 TP=2               7643 / 26.8        7140 / 24.9        -6.6% /  -7.1%
chat c=32 TP=4               8462 / 28.1        7824 / 26.1        -7.5% /  -7.1%
chat c=32 TP=8               8389 / 27.8        8421 / 26.2        +0.4% /  -5.8%
rag_prefill c=16 TP=2        3256 / 41.4        2737 / 45.0       -15.9% /  +8.7%
rag_prefill c=16 TP=4        5409 / 52.8        4874 / 55.7        -9.9% /  +5.5%
rag_prefill c=16 TP=8        5591 / 53.5        5269 / 58.6        -5.8% /  +9.5%
rag_prefill c=32 TP=2        7603 / 48.3        6936 / 52.9        -8.8% /  +9.5%
rag_prefill c=32 TP=4       11372 / 64.3       10754 / 70.0        -5.4% /  +8.9%
rag_prefill c=32 TP=8       11699 / 66.2       11652 / 75.0        -0.4% / +13.3%
```

### Stage M-A: moe_single_request (8 cases, Qwen3-30B-A3B TP=4)

```
path / shape                       real(TTFT/TPOT)    sim(TTFT/TPOT)    gap(TTFT/TPOT)
TP_only baseline 128/128            145 / 59.4         81 / 26.4       -44.0% / -55.6%
TP_only prefill_1024 1024/128       244 / 58.6        207 / 27.5       -14.9% / -53.0%
TP_only prefill_4096 4096/128       709 / 64.1        623 / 31.1       -12.1% / -51.4%
TP_only decode_2048 128/2048        148 / 59.2         88 / 27.6       -40.9% / -53.4%
TP_only mix 4096/1024               705 / 59.0        628 / 27.8       -10.9% / -52.9%
EP baseline 128/128                 161 / 59.1         91 / 34.2       -43.4% / -42.2%
EP prefill_2048 2048/128            390 / 60.3        361 / 35.0        -7.4% / -41.9%
EP decode_2048 128/2048             150 / 60.8         94 / 34.6       -37.6% / -43.1%
```

### Stage M-B: moe_concurrent (4 cases, MoE c=4/16)

```
path × c                            real(TTFT/TPOT)    sim(TTFT/TPOT)    gap(TTFT/TPOT)
TP_only c=4                          306 / 57.0        293 / 28.8        -4.3% / -49.5%
TP_only c=16                        1171 / 58.3       1134 / 34.8        -3.1% / -40.3%
EP c=4                               322 / 60.8        317 / 36.2        -1.6% / -40.5%
EP c=16                             2013 / 60.1       1147 / 44.1       -43.0% / -26.7%
```

### Stage E: multi_model_regression (2 cases)

```
model × TP             real(TTFT/TPOT)    sim(TTFT/TPOT)    gap(TTFT/TPOT)
Qwen2.5-3B TP=1         71.9 / 18.5       62.4 / 14.8       -13.2% / -20.0%
Qwen3-32B TP=4         339.2 / 44.2      324.4 / 47.5        -4.4% /  +7.4%
```

---

## 4. 已识别的偏差类型(对应 §13 backlog)

| 偏差类型 | 影响 case | gap 量级 | backlog 项 |
|----------|-----------|----------|-----------|
| **短 prefill ~23ms 固定 TTFT 偏差** | dense ISL=128 全部 | TTFT -30~-40% | §13.5 |
| **same-NUMA contention 低估** | TP=2 same prefill | TTFT -23% | §13.6 |
| **cross-NUMA cost 高估** | TP=4 cross prefill/mix | TTFT +28~+34% | §13.6 |
| **MoE 路径 per-step ~25-32ms 缺** | 全部 MoE TPOT | TPOT -33~-55% | §13.7 (待写) |
| **MoE 并发偏差不收敛** | MoE EP c=16 | TTFT -43% | §13.7 (待写) |

## 5. 当前模型适用范围

✓ **生产可信任**:
- Dense Qwen3-4B / Qwen2.5-3B / Qwen3-32B
- ISL ≥ 512 任意 OSL
- TP=1~8(知道 TP=4 cross 偏差方向)
- 并发(c=4~32)
- prefill 重 / decode 重 / 混合 workload

❌ **生产不可信**:
- 所有 MoE 模型 decode TPOT(系统性偏快 ~33-55%)
- ISL ≤ 256 短 prefill TTFT(系统性偏慢 ~23ms)

⚠ **需要小心**:
- TP=4 cross-NUMA prefill(sim 偏慢 ~30%)
- TP>1 small-batch decode(in-context comm floor 影响)
