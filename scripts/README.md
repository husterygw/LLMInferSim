# scripts/

## `bench_compare.sh`

真 GPU vs LLMInferSim 仿真 TTFT/TPOT 对比 bench.

### 用法

```bash
# 默认: Qwen3-4B-Instruct-2507 on RTX 4090
bash scripts/bench_compare.sh

# 自定义模型 / 硬件
MODEL=/path/to/Qwen3-32B HW=H100 bash scripts/bench_compare.sh

# 多模型对比 (逗号分隔)
MODELS=/data/ygw/models/Qwen3-4B-Instruct-2507,/data/ygw/models/Qwen2.5-3B-Instruct \
    bash scripts/bench_compare.sh

# 调度参数 sweep
MAX_NUM_SEQS=64 MAX_BATCH_TOKENS=4096 bash scripts/bench_compare.sh
```

### 流程

1. 拉一个**真 GPU** vLLM server(port 8810)→ 跑 5 个 scenario `vllm bench serve` → 落 `real_*.txt`
2. 拉一个**仿真** vLLM server(`VLLM_VIRTUAL_BACKEND=1`, port 8811)→ 跑同样 5 个 scenario → 落 `sim_*.txt`
3. 提取 `Mean TTFT` / `Mean TPOT`,打印对比表(含 gap%)

### Scenarios

| name | input | output | num_prompts |
|------|------:|-------:|------------:|
| short_short | 128 | 16 | 20 |
| med_short | 512 | 16 | 20 |
| long_short | 2048 | 16 | 10 |
| short_long | 128 | 128 | 10 |
| med_med | 512 | 64 | 20 |

### 环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `MODELS` | `$MODEL`(单模型) | 多模型路径,**逗号分隔**,每个模型独立一组 server pair + 5 scenario + summary |
| `MODEL` | `/data/ygw/models/Qwen3-4B-Instruct-2507` | 单模型路径 / HF id(`MODELS` 没设时用) |
| `HW` | `RTX_4090` | `LLM_INFER_SIM_HW` 硬件 profile |
| `CONDA_ENV` | `llm_sim` | conda 环境(空跳过激活) |
| `CUDA_VISIBLE_DEVICES` | `0` | |
| `RESULTS_DIR` | `/tmp/bench_compare_results` | 输出目录 |
| `GPU_MEM_UTIL` | `0.5` | `--gpu-memory-utilization` |
| `MAX_MODEL_LEN` | `4096` | `--max-model-len` |
| `MAX_NUM_SEQS` | `16` | `--max-num-seqs` |
| `MAX_BATCH_TOKENS` | `8192` | `--max-num-batched-tokens` |
| `PREFIX_CACHE` | `off` | `on` 开 prefix cache(默认关,避免污染对比) |
| `ENFORCE_EAGER` | `on` | `off` 开 CUDA Graph(默认 eager,跟 sim 的 module 级 kernel_overhead 语义对齐) |
| `TP` | `1` | `--tensor-parallel-size`,大模型需要多卡时设(默认自动 `CUDA_VISIBLE_DEVICES=0,...,TP-1`) |

### 预期输出

```
Comparison: REAL vs SIM
  model      = /data/ygw/models/Qwen3-4B-Instruct-2507
  hw (sim)   = RTX_4090
  prefix     = off
================================================================
scenario           real_TTFT     sim_TTFT     gap%      real_TPOT     sim_TPOT     gap%
--------------------------------------------------------------------
short_short        421.87       275.45    -34.7%        20.36        18.76     -7.9%
med_short          571.38       539.16     -5.6%        21.10        20.33     -3.6%
long_short         865.28       812.46     -6.1%        39.08        45.33    +16.0%
short_long         124.03       109.98    -11.3%        19.90        18.41     -7.5%
med_med            754.80       713.38     -5.5%        20.38        20.34     -0.2%
```

### 前置条件

- 真机侧:需 1 张能跑目标模型的 GPU(对 Qwen3-4B-BF16 来说 RTX 4090 / 24GB 即可)
- 模型权重已 download(`HF_HUB_OFFLINE=1` 强制 offline)
- conda env `llm_sim` 安装好 vLLM + LLMInferSim(`pip install -e .` from repo root)
- 跑一次 5-7 分钟(真机 / 仿真各拉一次 server,7 个 bench 命令)
