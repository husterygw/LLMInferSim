"""MetricsCollector — 详设 §4.10.1。

阶段 3 设计要点:
  - 时间一律用 **simulator time** (累加 cost.total_latency), 不用 wall-clock。
    instant TimeEmulator 模式下 wall-clock 几乎为零, 用它会污染指标。
  - per-request 生命周期: arrival → first_token → 后续 token → completion
    * arrival_time 在 request 第一次出现时记录
    * first_token 在该 request 第一次 generated_tokens >= 1 的 step 上记录
    * per_token_latencies 累加每个调度 step 的 cost.total_latency
    * completion_time 在 request_id 进入 SchedulerOutput.finished_req_ids 时记录
  - step-level breakdown 记录: total_latency, compute_time, memory_time, comm_time
    + bottleneck / phase / num_tokens / batch / sim_time。
"""
from __future__ import annotations

from typing import Iterable

from llm_infer_sim.core.cost_model.cost_result import GlobalStepCost
from llm_infer_sim.core.workload.request_state import RequestMetrics, SystemMetrics
from llm_infer_sim.core.workload.workload import GlobalStepWorkload, StepPhase


def _percentile(data: list[float], p: float) -> float:
    """简单 percentile 实现, 不依赖 numpy。"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    if len(sorted_data) == 1:
        return sorted_data[0]
    idx = (p / 100.0) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


class MetricsCollector:
    """收集 step + request 级别的指标。"""

    def __init__(self) -> None:
        self.step_records: list[dict] = []
        self.requests: dict[str, RequestMetrics] = {}
        self.sim_time: float = 0.0   # 累加的 simulator time (秒)

    # ------- 主接口 -------

    def record_step(
        self,
        workload: GlobalStepWorkload,
        cost: GlobalStepCost,
        finished_req_ids: Iterable[str] = (),
    ) -> None:
        """记录一个 step。

        约定:
          - sim_time 在 step 调度发生时取当前值 (即"step 开始时间")
          - per_token_latencies 把本 step 的 cost.total_latency 计入所有
            被调度到的 request (mixed 时多个 request 共享 step 时间, 这是
            serving simulator 的近似)
          - first_token: 当 request 在本 step 进入 DECODE phase (即 prefill 完成)
            或 generated_tokens 在本 step 增加到 ≥1, 记录 first_token_time
            = sim_time + cost.total_latency (即"first token 出炉时间")
          - completion: finished_req_ids 中的 req_id 在本 step 末尾完成
        """
        step_start_sim = self.sim_time
        step_end_sim = self.sim_time + cost.total_latency

        self.step_records.append({
            "step_id": cost.step_id,
            "phase": cost.phase,
            "total_latency": cost.total_latency,
            "compute_time": cost.compute_time,
            "memory_time": cost.memory_time,
            "comm_time": cost.comm_time,
            "num_tokens": workload.total_scheduled_tokens,
            "batch_size": workload.batch_size,
            "sim_time_start": step_start_sim,
            "sim_time_end": step_end_sim,
        })

        for req in workload.requests:
            rm = self.requests.get(req.request_id)
            if rm is None:
                rm = RequestMetrics(
                    request_id=req.request_id,
                    arrival_time=step_start_sim,
                    target_output_len=req.target_output_len,
                )
                self.requests[req.request_id] = rm

            rm.per_token_latencies.append(cost.total_latency)

            # first_token: 进入 DECODE 或 generated_tokens 已 >=1 都说明 prefill 已完成
            # 这个 step 之后的 sim_time_end 就是 first token 出炉时间
            if not rm.saw_first_token:
                # 在 DECODE phase 时表示 prefill 已经在前面 step 完成 → first_token_time
                # 可能在更早 step (但我们没记录), 用当前 step_end 作近似 fallback
                # 在 PREFILL phase 中 generated_tokens 通常 = 0, 但若 prompt 完成且
                # 本 step 是首个产出 token 的 step, generated_tokens 会变成 1
                phase = req.phase
                produced_first = (req.generated_tokens >= 1) or (phase == StepPhase.DECODE)
                # 真正"本 step 完成 prefill 第一次产 token"的判定:
                # PREFILL/CHUNKED_PREFILL phase 下若 num_tokens 已经把整个 prompt
                # 啃完 (即 context_len + new tokens >= prompt_len 是 vllm 内部的事,
                # 我们这里用 generated_tokens 上升或 phase 转 DECODE 当信号)
                if produced_first:
                    rm.first_token_time = step_end_sim
                    rm.saw_first_token = True

            rm.output_tokens = max(rm.output_tokens, req.generated_tokens)
            if req.target_output_len > rm.target_output_len:
                rm.target_output_len = req.target_output_len

        # 处理本 step 完成的 request
        for fid in finished_req_ids:
            rm = self.requests.get(fid)
            if rm is None or rm.completed:
                continue
            rm.completion_time = step_end_sim
            rm.completed = True

        self.sim_time = step_end_sim

    # ------- 聚合 -------

    def get_summary(self) -> SystemMetrics:
        completed = [r for r in self.requests.values() if r.completed]

        ttfts = [r.ttft for r in completed if r.saw_first_token]
        tpots = [r.tpot for r in completed if len(r.per_token_latencies) > 1]
        e2es = [r.e2e_latency for r in completed]

        total_output_tokens = sum(r.output_tokens for r in completed)
        elapsed = self.sim_time

        return SystemMetrics(
            total_requests=len(self.requests),
            completed_requests=len(completed),
            total_output_tokens=total_output_tokens,
            total_steps=len(self.step_records),
            elapsed_sim_time=elapsed,
            requests_per_second=(len(completed) / elapsed) if elapsed > 0 else 0.0,
            output_tokens_per_second=(total_output_tokens / elapsed) if elapsed > 0 else 0.0,
            avg_ttft=(sum(ttfts) / len(ttfts)) if ttfts else 0.0,
            p50_ttft=_percentile(ttfts, 50),
            p90_ttft=_percentile(ttfts, 90),
            p99_ttft=_percentile(ttfts, 99),
            avg_tpot=(sum(tpots) / len(tpots)) if tpots else 0.0,
            p50_tpot=_percentile(tpots, 50),
            p90_tpot=_percentile(tpots, 90),
            p99_tpot=_percentile(tpots, 99),
            avg_e2e=(sum(e2es) / len(e2es)) if e2es else 0.0,
            p50_e2e=_percentile(e2es, 50),
            p90_e2e=_percentile(e2es, 90),
            p99_e2e=_percentile(e2es, 99),
            avg_step_breakdown=self._avg_step_breakdown(),
        )

    def _avg_step_breakdown(self) -> dict[str, float]:
        if not self.step_records:
            return {}
        keys = ["compute_time", "memory_time", "comm_time", "total_latency"]
        totals = {k: 0.0 for k in keys}
        for s in self.step_records:
            for k in keys:
                totals[k] += s.get(k, 0.0)
        n = len(self.step_records)
        return {k: v / n for k, v in totals.items()}
