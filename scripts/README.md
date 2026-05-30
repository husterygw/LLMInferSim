# scripts/

可执行工具目录,按用途分子目录:

```text
bench/    bench_cases.py / bench_compare.sh / run_bench_*.sh / run_stage_bench.sh / analyze_bench.py
report/   report_*_gap.py / analyze_collectives.py / analyze_sim_step_latency.py / analyze_stage.py / analyze_vllm_worker_profile.py
measure/  measure_*.py/.sh + run_collective_sweep.sh(标定实测)
profile/  profile_moe_fused_measured.py / profile_virtual_runner.py / profile_vllm_runtime_overhead.py
debug/    debug_qwen3_30b_offline.py / explain_op_db_miss.py / step0_ar_param_sweep.py
lib/      _extract_metrics.py / _meas_common.py(共享 helper,被上面脚本 import)
```

工具**行为的断言**放在 `tests/tools/`(`tools` marker),不放在本目录,避免测试逻辑
污染工具。这些工具默认不进 PR gate;测试分层与默认 gate 命令见 `tests/README.md`。

## Case-Driven Benchmarks

`bench_cases.py` is the single source of truth for benchmark matrices. It
generates JSONL cases; `bench_compare.sh` only executes those cases against real
vLLM and LLMInferSim virtual backend.

### Common Commands

```bash
bash scripts/bench/run_bench_suite.sh single_tp1_roofline
bash scripts/bench/run_bench_suite.sh batch_tp1_sweep
bash scripts/bench/run_bench_suite.sh tp_comm_sweep
bash scripts/bench/run_bench_suite.sh tp_batch_sweep
bash scripts/bench/run_bench_suite.sh long_context_sweep
bash scripts/bench/run_bench_suite.sh moe_tp_sweep
bash scripts/bench/run_bench_suite.sh moe_ep_sweep
bash scripts/bench/run_bench_suite.sh multi_model_regression
```

Legacy aliases are still accepted:

```text
A -> single_tp1_roofline
B -> tp_comm_sweep
C -> batch_tp1_sweep
D -> tp_batch_sweep
E -> multi_model_regression
```

### Generate Cases

```bash
python scripts/bench/bench_cases.py --suite single_tp1_roofline --out /tmp/cases.jsonl
python scripts/bench/bench_cases.py --suite batch_tp1_sweep --out /tmp/cases.jsonl
python scripts/bench/bench_cases.py --list
```

### Execute Cases

```bash
bash scripts/bench/bench_compare.sh --cases /tmp/cases.jsonl --out /tmp/bench_out
bash scripts/bench/bench_compare.sh --cases /tmp/cases.jsonl --out /tmp/bench_out --dry-run
```

`bench_compare.sh` does not define scenarios, batch sizes, or ISL/OSL matrices.
Those belong in `bench_cases.py`.

Default case policy:

```text
prefix_cache=off
chunked_prefill=off
max_num_seqs=None
num_warmups=1
```

Generated suites set `max_model_len` and `max_num_batched_tokens` explicitly
when vLLM needs a bounded serving length. For Qwen3-4B suites this avoids
starting from the model intrinsic 262K context and failing vLLM's scheduler
check.

So the executor explicitly passes:

```text
--no-enable-prefix-caching
--no-enable-chunked-prefill
```

and only passes `--max-model-len` / `--max-num-seqs` when the case sets them.

### Analyze

```bash
python scripts/bench/analyze_bench.py /tmp/llm_infer_sim_bench --suite single_tp1_roofline
python scripts/bench/analyze_bench.py /tmp/llm_infer_sim_bench --csv /tmp/bench.csv
```
