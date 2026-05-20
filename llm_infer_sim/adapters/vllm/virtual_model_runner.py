"""VirtualModelRunner — 阶段 2: 接 llm-viewer builder。

阶段 2 增量 (vs 阶段 1):
  - 用 adapters/vllm/profile_extractor.extract_profile_bundle 一次构造 ProfileBundle
    (ModelConfig + LegacyDeployConfig + HardwareConfig + EfficiencyProfile)
  - 弃用阶段 1 手写的 model_meta 抽取 (只 7 个字段, 漏 head_dim explicit / MLA / MoE)
  - ModelCoreCostModel 改接 ProfileBundle, 内部走 llm-viewer dense/moe_layer_time

阶段 3.5 重构:
  - ProfileManager.from_vllm_config 改名为 extract_profile_bundle, 实现搬到
    adapters/vllm/profile_extractor.py (core 完全框架无关, 详设 §1.1)

阶段 3+ 起:
  - chunked prefill mixed step (MixedAttentionEstimator)
  - 阶段 4: TP cost 聚合
  - 阶段 5/6/8: 自动 (因为 dense/moe_layer_time 已支持 MoE / MLA / V4)
"""
from __future__ import annotations

import os
import sys
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.outputs import ModelRunnerOutput

from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
from llm_infer_sim.adapters.vllm.step_extractor import VllmStepExtractor
from llm_infer_sim.core.cost_model.cost_result import GlobalStepCost
from llm_infer_sim.core.cost_model.model_core import ModelCoreCostModel
from llm_infer_sim.core.metrics.breakdown import format_step_breakdown
from llm_infer_sim.core.metrics.collector import MetricsCollector
from llm_infer_sim.core.metrics.reporter import ReportGenerator
from llm_infer_sim.core.ops.kv_transfer import kv_transfer_time
from llm_infer_sim.core.simulation.kv_block_allocator import KVBlockAllocator
from llm_infer_sim.core.simulation.output_generator import FakeTokenGenerator
from llm_infer_sim.core.simulation.time_emulator import VirtualTimeEmulator
from llm_infer_sim.core.workload.workload import GlobalStepWorkload


def _log(msg: str) -> None:
    print(f"[VirtualModelRunner] {msg}", file=sys.stderr, flush=True)


class VirtualModelRunner:
    """阶段 2: extract → llm-viewer builder cost → realtime sleep → fake token。"""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

        # ---- 1. ProfileBundle (ModelConfig + LegacyDeployConfig + hw + efficiency) ----
        self.bundle = extract_profile_bundle(vllm_config)

        # ---- 2. cost model (走 llm-viewer dense/moe_layer_time) ----
        self.model_cost = ModelCoreCostModel(self.bundle)

        # ---- 3. time emulator (realtime by default) ----
        mode = os.environ.get("LLM_INFER_SIM_TIME_MODE", "realtime")
        self.time_emulator = VirtualTimeEmulator(mode=mode)

        # ---- 4. request state cache (cached step 没 sampling_params) ----
        self._request_states: dict[str, dict] = {}

        # ---- 4b. metrics (阶段 3 D 块) ----
        self.metrics = MetricsCollector()

        # ---- 5. op dump 控制 ----
        # LLM_INFER_SIM_DUMP_OPS: 0=不打 | 1=只打首个有效 step | 2=每个 step 都打
        self._dump_ops_mode = int(os.environ.get("LLM_INFER_SIM_DUMP_OPS", "0"))
        self._ops_dumped = False

        # ---- 6. per-request 调度 dump 控制 ----
        # LLM_INFER_SIM_DUMP_REQUESTS: 0=不打 | 1=每个 step 列出所有 request 的
        # req_id/phase/num_tokens/ctx_len/generated_tokens
        # vLLM v1 scheduler 自己默认不打 step-level 调度决策, 我们在 step_extractor
        # 已经抓到了完整 per-request 信息, 这里负责打。
        self._dump_requests_mode = int(
            os.environ.get("LLM_INFER_SIM_DUMP_REQUESTS", "0")
        )

        # ---- 7. fake token generator (详设 §4.3.5) ----
        # fixed (默认): 所有 token = 1, 兼容阶段 0/2 e2e。
        # deterministic_hash: token 由 (prompt_token_ids + num_generated) md5 决定,
        # 让重复 prompt 在 prefix caching ON 下命中 "prompt + 输出" 完整链。
        self.fake_token_gen = FakeTokenGenerator.from_env(
            vocab_size=self.bundle.model.vocab_size
        )

        # ---- 8. KV block allocator (详设 §10.5 4.5 + §7.6) ----
        # lazy-init: 首个 execute_step 时构造, 因为 num_gpu_blocks 在 worker
        # initialize_from_config 之后才确定 (我们 runner 不直接看 KVCacheConfig)。
        self._block_allocator: KVBlockAllocator | None = None

        # ---- 9. PD 分离 (详设 §7.6) ----
        self._pd_cfg = self.bundle.deploy.pd
        self._pd_total_transfer_time: float = 0.0   # 累计 (供 aggregate 报告)
        self._pd_total_transfer_bytes: int = 0
        self._pd_num_transfers: int = 0
        # 记录 producer 上已经 send 过 KV 的 req_id (防止 chunked prefill 多次 send);
        # consumer 上记录已经 recv 过的 req_id (防止重复 recv)。
        self._pd_handled_reqs: set[str] = set()

        _log(
            f"init model={self.bundle.model.name} {self._model_summary()} "
            f"hw={self.bundle.hw.name if hasattr(self.bundle.hw, 'name') else '?'} "
            f"time_mode={mode} dump_ops={self._dump_ops_mode} "
            f"dump_requests={self._dump_requests_mode} "
            f"fake_token_mode={self.fake_token_gen.mode}"
        )

    # ------- public -------

    def execute_step(
        self,
        scheduler_output: Any,
        step_id: int,
        rank: int = 0,
    ) -> ModelRunnerOutput:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        if num_tokens == 0:
            return self._empty_output()

        self._update_request_states(scheduler_output)

        workload = VllmStepExtractor.extract(
            scheduler_output, step_id, self._request_states,
        )

        cost = self._estimate_cost(workload)

        # ---- KV block allocator: 跟踪每 step alloc/free/dedup ----
        block_stats = self._step_block_allocator(
            scheduler_output, workload.num_prefix_cached_tokens
        )

        # ---- PD 分离: producer/consumer 的 KV 传输 cost (§7.6) ----
        pd_extra_time = 0.0
        pd_extra_bytes = 0
        if self._pd_cfg.enabled and self._block_allocator is not None:
            pd_extra_time, pd_extra_bytes = self._compute_pd_transfer_cost(
                scheduler_output
            )
            if pd_extra_time > 0:
                cost = GlobalStepCost(
                    step_id=cost.step_id,
                    phase=cost.phase,
                    total_latency=cost.total_latency + pd_extra_time,
                    compute_time=cost.compute_time,
                    memory_time=cost.memory_time,
                    comm_time=cost.comm_time + pd_extra_time,
                    per_layer=cost.per_layer,
                )

        step_line = format_step_breakdown(cost)
        if workload.num_prefix_cached_tokens > 0:
            step_line += (
                f" | cached={workload.num_prefix_cached_tokens} "
                f"computed={workload.total_scheduled_tokens}"
            )
        if block_stats is not None and block_stats.blocks_dedup_hit > 0:
            step_line += (
                f" | blocks_dedup={block_stats.blocks_dedup_hit} "
                f"new={block_stats.new_blocks_allocated} "
                f"in_use={block_stats.blocks_in_use_after}/"
                f"{self._block_allocator.num_blocks_total}"
            )
        if pd_extra_time > 0:
            step_line += (
                f" | pd_{self._pd_cfg.role}={pd_extra_bytes/1e6:.2f}MB "
                f"+{pd_extra_time*1e3:.2f}ms"
            )
        # ---- DP 同步 (详设 §10.5 G3): step latency = max(per-rank cost) ----
        # vLLM v1 DP 用 padding token 强同步, 慢者拖快者. 没这一步, 各 rank 独立 sleep
        # 会让 batch 不均时 sim 偏快。dp_size=1 时跳过。
        synced_latency = self._sync_dp_latency(cost.total_latency)
        if synced_latency != cost.total_latency:
            step_line += (
                f" | dp_sync max={synced_latency*1e3:.2f}ms "
                f"(local={cost.total_latency*1e3:.2f}ms)"
            )
            # 把同步后的 max latency 反写回 cost, 让 metrics 也按 max 算
            cost = GlobalStepCost(
                step_id=cost.step_id,
                phase=cost.phase,
                total_latency=synced_latency,
                compute_time=cost.compute_time,
                memory_time=cost.memory_time,
                comm_time=cost.comm_time,
                per_layer=cost.per_layer,
            )

        _log(step_line)
        self._maybe_dump_requests(workload)

        # 记录到 metrics (sim time 在 collector 内累加)
        self.metrics.record_step(
            workload, cost,
            finished_req_ids=scheduler_output.finished_req_ids,
        )

        self.time_emulator.simulate(cost.total_latency)

        return self._build_model_runner_output(scheduler_output)

    def _sync_dp_latency(self, local_latency: float) -> float:
        """跨 DP rank 取 max latency (详设 §10.5 G3).

        vLLM v1 DP 各 rank 独立 scheduler, 但每 step 用 padding-token 同步:
        慢 rank 决定整 step 时长。这里调 vLLM 的 dp_group 做 all_reduce(MAX, gloo)。

        Fast paths:
          - dp_size == 1 → no-op (大部分 example)
          - vLLM PG 尚未初始化 → no-op (worker init 之前)
          - 任何异常 → no-op + 警告一次 (不影响 cost 主路径)
        """
        dp_size = getattr(self.vllm_config.parallel_config, "data_parallel_size", 1) or 1
        if dp_size <= 1:
            return local_latency
        try:
            from vllm.distributed.parallel_state import get_dp_group
            import torch

            dp_group = get_dp_group()
            if dp_group is None or dp_group.world_size <= 1:
                return local_latency
            t = torch.tensor([local_latency], dtype=torch.float64)
            # all_reduce MAX 在 gloo 上是支持的
            torch.distributed.all_reduce(
                t, op=torch.distributed.ReduceOp.MAX, group=dp_group.device_group,
            )
            return float(t.item())
        except Exception as e:  # pragma: no cover
            if not getattr(self, "_dp_sync_warned", False):
                _log(f"WARN: DP sync 失败, fallback 到 local latency: {type(e).__name__}: {e}")
                self._dp_sync_warned = True
            return local_latency

    def get_report(self) -> ReportGenerator:
        """返回 ReportGenerator 供 example/test 在 generate 完后输出报告。"""
        return ReportGenerator(self.metrics, block_allocator=self._block_allocator)

    def _compute_pd_transfer_cost(self, scheduler_output: Any) -> tuple[float, int]:
        """检测本 step 内的 PD KV 传输事件, 返回 (extra_latency_s, extra_bytes).

        Producer (kv_producer / kv_both):
            for each req: 若本 step 完成 prefill (computed_after >= prompt_len)
            且尚未 send → 计 kv_send cost = req_kv_bytes / bandwidth + latency.

        Consumer (kv_consumer / kv_both):
            for each new_req: 若 num_computed_tokens == prompt_len (vLLM 表示
            prefill 已在外部完成) 且尚未 recv → 计 kv_recv cost.
        """
        if self._block_allocator is None or not self._pd_cfg.enabled:
            return 0.0, 0
        bw = self._pd_cfg.resolve_bandwidth()
        lat_us = self._pd_cfg.resolve_latency_us()
        total_time = 0.0
        total_bytes = 0

        # producer 路径: 检 new_req 本 step 完成
        if self._pd_cfg.is_producer:
            for new_req in scheduler_output.scheduled_new_reqs:
                rid = new_req.req_id
                if rid in self._pd_handled_reqs:
                    continue
                prompt_len = len(new_req.prompt_token_ids or [])
                ntok = scheduler_output.num_scheduled_tokens.get(rid, 0)
                computed_after = new_req.num_computed_tokens + ntok
                if computed_after >= prompt_len > 0:
                    bytes_to_send = self._block_allocator.req_kv_bytes(rid)
                    t = kv_transfer_time(bytes_to_send, bw, lat_us)
                    total_time += t
                    total_bytes += bytes_to_send
                    self._pd_handled_reqs.add(rid)

            # 也检 cached_req (chunked prefill 续段最后一步)
            cached = scheduler_output.scheduled_cached_reqs
            for i, rid in enumerate(cached.req_ids):
                if rid in self._pd_handled_reqs:
                    continue
                # prompt_len 从 state 拿
                state = self._request_states.get(rid)
                if state is None:
                    continue
                prompt_len = len(state.get("prompt_token_ids", []))
                if prompt_len == 0:
                    continue
                ntok = scheduler_output.num_scheduled_tokens.get(rid, 1)
                # cached_req 在 chunked prefill 中 ntok > 1; decode 时 ntok==1.
                # 仅当 ntok > 1 且 prefill 收尾 才算 send 事件
                if ntok <= 1:
                    continue
                computed_after = cached.num_computed_tokens[i] + ntok
                if computed_after >= prompt_len:
                    bytes_to_send = self._block_allocator.req_kv_bytes(rid)
                    t = kv_transfer_time(bytes_to_send, bw, lat_us)
                    total_time += t
                    total_bytes += bytes_to_send
                    self._pd_handled_reqs.add(rid)

        # consumer 路径: new_req with num_computed == prompt_len
        if self._pd_cfg.is_consumer:
            for new_req in scheduler_output.scheduled_new_reqs:
                rid = new_req.req_id
                if rid in self._pd_handled_reqs:
                    continue
                prompt_len = len(new_req.prompt_token_ids or [])
                if prompt_len > 0 and new_req.num_computed_tokens >= prompt_len:
                    # prefill 已在外部完成, 此进程需 recv 完整 KV
                    # bytes = ceil(prompt_len / block_size) × block_bytes
                    blocks = -(-prompt_len // self._block_allocator.block_size)
                    bytes_to_recv = blocks * self._block_allocator.block_bytes
                    t = kv_transfer_time(bytes_to_recv, bw, lat_us)
                    total_time += t
                    total_bytes += bytes_to_recv
                    self._pd_handled_reqs.add(rid)

        # 清理 finished_req_ids 的状态 (释放 _pd_handled_reqs)
        for fid in scheduler_output.finished_req_ids:
            self._pd_handled_reqs.discard(fid)

        self._pd_total_transfer_time += total_time
        self._pd_total_transfer_bytes += total_bytes
        if total_time > 0:
            self._pd_num_transfers += 1
        return total_time, total_bytes

    def get_pd_stats(self) -> dict:
        """供报告 / rpc 拉 PD 累计统计."""
        return {
            "pd_enabled": self._pd_cfg.enabled,
            "pd_role": self._pd_cfg.role,
            "pd_connector": self._pd_cfg.connector_name,
            "pd_bandwidth_gbps": self._pd_cfg.resolve_bandwidth() if self._pd_cfg.enabled else 0.0,
            "pd_total_transfer_time_s": self._pd_total_transfer_time,
            "pd_total_transfer_bytes": self._pd_total_transfer_bytes,
            "pd_num_transfers": self._pd_num_transfers,
        }

    def _step_block_allocator(self, scheduler_output: Any, num_cached_tokens: int):
        """lazy-init allocator (首次拿到 num_gpu_blocks) 后逐 step 调 .step()."""
        if self._block_allocator is None:
            cc = getattr(self.vllm_config, "cache_config", None)
            if cc is None:
                return None
            num_blocks = getattr(cc, "num_gpu_blocks", None) or 0
            block_size = getattr(cc, "block_size", 16) or 16
            if num_blocks <= 0:
                return None
            self._block_allocator = KVBlockAllocator(
                model=self.bundle.model,
                block_size=block_size,
                num_blocks_total=num_blocks,
                kv_byte=self.bundle.deploy.kv_byte,
            )
            _log(
                f"KVBlockAllocator init: block_size={block_size} "
                f"num_blocks_total={num_blocks} "
                f"block_bytes={self._block_allocator.block_bytes} "
                f"(kv_byte={self.bundle.deploy.kv_byte})"
            )
        return self._block_allocator.step(scheduler_output, num_cached_tokens)

    # ------- internals -------

    def _estimate_cost(self, workload: GlobalStepWorkload) -> GlobalStepCost:
        result = self.model_cost.estimate(workload)
        self._maybe_dump_ops(workload, result)
        return GlobalStepCost(
            step_id=workload.step_id,
            phase=workload.phase.value,
            total_latency=result["total_time"],
            compute_time=result["compute_time"],
            memory_time=result["memory_time"],
            comm_time=result["comm_time"],
            per_layer=result.get("per_layer", []),
        )

    def _maybe_dump_requests(self, workload: GlobalStepWorkload) -> None:
        """每 step per-request 调度细节: req_id / phase / num_tokens / ctx_len / generated."""
        if self._dump_requests_mode == 0:
            return
        if not workload.requests:
            return
        _log(
            f"  -- step={workload.step_id} requests ({len(workload.requests)}): "
            f"prefill_req={workload.num_prefill_requests} "
            f"decode_req={workload.num_decode_requests}"
        )
        _log(
            f"     {'req_id':<10} {'phase':<18} "
            f"{'num_tok':>7} {'ctx_len':>7} {'gen_tok':>7} {'is_chunked'}"
        )
        for r in workload.requests:
            _log(
                f"     {r.request_id:<10} {r.phase.value:<18} "
                f"{r.num_tokens:>7} {r.context_len:>7} {r.generated_tokens:>7} "
                f"{r.is_chunked}"
            )

    def _maybe_dump_ops(self, workload: GlobalStepWorkload, result: dict) -> None:
        if self._dump_ops_mode == 0:
            return
        if self._dump_ops_mode == 1 and self._ops_dumped:
            return
        per_op = result.get("per_op", [])
        if not per_op:
            return
        _log(
            f"--- op dump @ step={workload.step_id} phase={workload.phase.value} "
            f"stage={result['stage']} tokens={result['tokens']} batch={result['batch']} "
            f"ctx_len={result['ctx_len']} (total {len(per_op)} ops) ---"
        )
        _log(
            f"  {'scope':<10} {'name':<20} {'category':<13} "
            f"{'flops':>10} {'mem_B':>10} "
            f"{'t_cmp(us)':>9} {'t_mem(us)':>9} {'t_tot(us)':>9} {'bound'}"
        )
        for r in per_op:
            _log(
                f"  {r['scope']:<10} {r['name']:<20} {r['category']:<13} "
                f"{r['flops']:>10.2e} {r['mem_bytes']:>10.2e} "
                f"{r['t_compute']*1e6:>9.2f} {r['t_memory']*1e6:>9.2f} "
                f"{r['t_total']*1e6:>9.2f} {r['bound']}"
            )
        _log(
            f"--- end op dump (sum t_compute={result['compute_time']*1e6:.2f}us "
            f"t_memory={result['memory_time']*1e6:.2f}us "
            f"t_comm={result['comm_time']*1e6:.2f}us "
            f"t_total={result['total_time']*1e6:.2f}us) ---"
        )
        self._ops_dumped = True

    def _model_summary(self) -> str:
        m = self.bundle.model
        head = (
            f"L={m.num_layers} H={m.hidden_dim} "
            f"n_h={m.num_heads} n_kv={m.num_kv_heads} "
            f"d={m.head_dim} ffn={m.ffn_dim} vocab={m.vocab_size}"
        )
        if m.is_moe:
            head += (
                f" MoE(E={m.num_experts} top_k={m.num_activated_experts} "
                f"expert_dim={m.expert_dim})"
            )
        if m.kv_lora_rank > 0:
            head += f" MLA(kv_lora_rank={m.kv_lora_rank})"
        return head

    def _update_request_states(self, scheduler_output: Any) -> None:
        for new_req in scheduler_output.scheduled_new_reqs:
            sp = new_req.sampling_params
            # 阶段 4.5: deterministic_hash 模式需要 prompt_token_ids 跨 step 可查。
            # fixed 模式也存, 内存代价低 (10K prompt × 4B int = 40KB/req); 简化分支。
            self._request_states[new_req.req_id] = {
                "target_output_len": (
                    int(sp.max_tokens) if sp is not None and sp.max_tokens else 0
                ),
                "prompt_token_ids": list(new_req.prompt_token_ids or []),
            }
        for fid in scheduler_output.finished_req_ids:
            self._request_states.pop(fid, None)

    def _build_model_runner_output(self, scheduler_output: Any) -> ModelRunnerOutput:
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        req_id_to_index = {rid: idx for idx, rid in enumerate(req_ids)}

        # 取每 req 当前已生成数 (本 step 之前): new_req → 0, cached_req → num_output_tokens
        cached = scheduler_output.scheduled_cached_reqs
        num_generated_by_id: dict[str, int] = {}
        for i, rid in enumerate(cached.req_ids):
            num_generated_by_id[rid] = cached.num_output_tokens[i]

        sampled_token_ids: list[list[int]] = []
        for rid in req_ids:
            state = self._request_states.get(rid, {})
            prompt_tokens = state.get("prompt_token_ids", [])
            num_gen = num_generated_by_id.get(rid, 0)
            tok = self.fake_token_gen.next_token(prompt_tokens, num_gen)
            sampled_token_ids.append([tok])
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    @staticmethod
    def _empty_output() -> ModelRunnerOutput:
        return ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
