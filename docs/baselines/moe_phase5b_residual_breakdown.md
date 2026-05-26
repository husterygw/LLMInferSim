# MoE Phase 5.B Calibration Residual Breakdown

> ## ⚠️ OBSOLETE (2026-05-26)
>
> 本文档原诊断 "+24% TPOT gap 主因在 scheduler/framework overhead, 不在 MoE compute" **是错的**.
>
> **真实 root cause**: `qwen.py:_build_moe_ffn_block` 把 vLLM 默认 EP path 当成 TRT-LLM/DeepEP 那种 AllToAll-based, 错加 2 个 AllToAll/layer × 48 layer. vLLM 默认 EP path 实际是 `fused_experts(expert_map=...)` + 单次 `tensor_model_parallel_all_reduce`, 跟 ep=1 tp>1 路径同 collective 形态.
>
> **fix 后实测** (2026-05-26 重跑 moe_ep_sweep):
>
> | case | Phase 1 baseline | fix 后 |
> |---|---|---|
> | i128_o128 tp4 ep4 | TPOT +19.87% | **-1.29%** |
> | i128_o2048 tp4 ep4 | TPOT +0.95% | **-11.51%** |
> | **driver i2048_o128 tp4 ep4** | **TPOT +24.27%** | **-0.81%** |
>
> 全部 ±15% 内, 不需要 calibration. Phase 5.A wiring 架构保留, calibration knob 全部回 neutral.
>
> 见 `memory: feedback_ep_collective_topology_vs_calibration.md`.
>
> 本文档以下章节保留作 historical reference, **不要按这套诊断 calibrate**.

---

**Date**: 2026-05-25 (obsolete 2026-05-26)
**Scope**: moe_plan §5.B validation — driver case `moe_ep_sweep i2048_o128 tp4 ep4`.

## 验收点 (plan §5.B step 5)

```
驱动 case TPOT: +24.27% → abs(TPOT_gap) ≤ 15%
```

**未达成**. Phase 5.B 校准 attempt 后 TPOT 反而恶化到 +32.36% (Δ +8.09%).

## 根因

| 维度 | 数据 | 含义 |
|---|---|---|
| **Phase 5 standalone MoE op gap** | `measured/roofline = 1.075` (driver bucket mean) | sim **偏快** 7.5% |
| **moe_ep_sweep 端到端 TPOT gap** | `(sim-real)/real = +24.27%` | sim **偏慢** 24% |
| **方向相反** | — | TPOT +24% gap **主因不在 MoE compute** |

## 关闭 MoEEfficiencyProfile (Phase 5.A wiring) 是否解决

- ✗ 不能. MoEEfficiencyProfile 三个 knob 只 cover MoE compute + MoEDispatch + moe_topk.
- TPOT +24% gap source 在 step 其它部分 (attention / scheduler async pipeline / collective overhead overlap).

## Residual breakdown estimate (engine smoke per-op 占比, TP=4 EP=4 decode, batch=1)

参考 Phase 3 engine smoke 数据 (`grep "TP=4 EP=4 decode"` in conversation):

```
moe_gate (router GEMM):       ~2us  ×  4 layers = ~8us
moe_topk:                      0us  ×  4        = 0us       (Phase 5.A wired, knob=0)
moe_dispatch_pre:              0us  ×  4        = 0us       (knob=0)
ep_alltoall_dispatch:        ~30us  ×  4        = ~120us
routed_experts:              ~75us  ×  4        = ~300us
ep_alltoall_combine:         ~30us  ×  4        = ~120us
moe_dispatch_post:             0us  ×  4        = 0us
attention / qkv / norm / o_proj:                  ~80us / step
Total step ≈ 635us, TPOT ≈ 635us
```

`real TPOT ≈ 635us × 1.24 ≈ 787us`. Gap = +152us / step.

MoE compute 部分占 step ~47% (296us). 即使 grouped_gemm_efficiency=0.5 让 routed_experts ×2 (+296us)，也只能 close 部分 gap，且会让 Phase 5 standalone gap 严重偏离 (single-op measurement 显示 sim 已经偏快, eff<1 让 sim 更快, 跟 standalone 实测一致 — 但 standalone 跟 end-to-end 矛盾)。

## 矛盾来源 (Phase 5 standalone vs end-to-end TPOT)

可能因素 (按 plan §2 "主要不足"):

1. **scheduler / async pipeline overlap**: vLLM v1 真实 scheduler 把 prefill / decode / comm 部分重叠, sim StepCostEngine 串行加. 真实 step 可以更长 (sim 没算重叠以外的 wait), real TPOT > sim TPOT.
2. **per-step framework overhead**: vLLM logits processor, sampling, request state mgmt 每 step 有 fixed overhead, sim 没 model. 这个量级几百 us per step, 跟 TPOT 24% gap (~150us @ 635us baseline) 量级吻合.
3. **CUDA graph capture overhead boundary**: real cudagraph step 含 capture trigger + graph launch overhead, sim 给 0.
4. **kv cache management**: real vLLM block table 更新 / paged-attention indexing 每 step, sim 不算.

这些都不在 MoE Phase 5.B knob 覆盖范围内.

## 决策

1. **Phase 5.A wiring 保留** (架构完成, knob 准备好接 future calibration).
2. **Phase 5.B v1 calibration values 回到中性** (profile_id="rtx_4090_phase5b_neutral", knob 全 0/1).
3. **+24% gap follow-up** 需要在更广 phase 处理:
   - dense attention / scheduler / framework_overhead recalibration
   - 跟 Step 5 dense calibration (`prefill_worker_overhead_s=5ms`) 类型化扩展到 decode step overhead
   - 不属于 moe_plan 范围, 在新计划

## Phase 5.B 实际产出

```
✓ Phase 5.A wiring 架构完整: MoEEfficiencyProfile + 3 knobs + RooflineBackend 后处理 + metadata
✓ rtx_4090_moe_efficiency_v1 placeholder profile_id 标记可 follow-up
✓ Residual breakdown 报告写明 +24% TPOT gap 不在 MoE knob 覆盖范围
✗ TPOT ±15% 数值收敛未达 (根因不在 MoE 部分)
```

## 留 follow-up

- 跑 `scripts/measure_moe_decode_profile.py` 实测 vLLM Qwen3-30B-A3B kernel breakdown,
  确认 vLLM 真实 moe_align/sort/gather kernel 数量 + latency.
- 把 calibration 扩到 bucket table (`moe_grouped_gemm_efficiency[phase][tokens_bucket]`),
  分别 fit standalone gap 而非 single value.
- 独立 phase 校准 dense attention + scheduler overhead — 这跟 moe_plan §6 collective
  follow-up 应同时考虑.
