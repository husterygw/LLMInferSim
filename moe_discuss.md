# MoE Fused Kernel Gap Discussion

Date: 2026-05-26

This note summarizes the MoE performance discussion around the standalone vLLM
`fused_experts` case:

```text
TP1 EP4 balanced
num_tokens = 2048
hidden = 2048
moe_intermediate = 768
topk = 8
num_experts = 128
local_experts = num_experts / ep = 32
tokens_per_device = num_tokens * topk / ep = 4096
avg_rows_per_expert = tokens_per_device / local_experts = 128
dtype = bf16
device = RTX 4090
vLLM = 0.19.1
```

## Original Gap

Baseline row:

```text
case_id:
  moe__moe_n2048_motp1_moep4_balanced_cudagraph__88d519e81af8

measured = 642.048 us
roofline = 332.881 us
gap      = 1.93x
```

This is large enough that treating MoE as one opaque `routed_experts` roofline
term hides the important source of error.

## Existing Roofline Meaning

Current `routed_experts` roofline includes:

```text
expert_flops
expert_weight_read
expert_act_load
expert_act_store
```

For this case the reconstructed roofline terms are:

```text
t_compute_us       = 233.987
t_memory_us        = 332.881
modeled_weight_mb  = 301.990
modeled_act_io_mb  = 33.554
modeled_total_mb   = 335.544
```

The roofline bottleneck is memory. The initial diagnostic residual is:

```text
residual_us       = measured - roofline = 309.167 us
residual_fraction = 48.15%
extra_mem_equiv   = 311.640 MB at RTX 4090 modeled bandwidth
```

The logical gate/up/down split of the roofline compute term is:

```text
gate_compute_us = 77.996
up_compute_us   = 77.996
down_compute_us = 77.996
```

This split is only logical attribution. vLLM may fuse gate/up and does not
necessarily expose three independent GEMM kernels.

Diagnostic script:

```bash
python scripts/report_moe_internal_breakdown.py \
  --hardware RTX_4090 \
  --framework vllm \
  --framework-version 0.19.1 \
  --case-id moe__moe_n2048_motp1_moep4_balanced_cudagraph__88d519e81af8 \
  --csv /tmp/moe_internal.csv
```

## Measured Kernel Breakdown

To break down the measured time itself, we profiled the standalone vLLM
`fused_experts` path with `torch.profiler` in conda env `llm_sim`, using GPU 0.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n llm_sim python scripts/profile_moe_fused_measured.py \
  --iters 10 \
  --warmup 5 \
  --trace /tmp/moe_fused_tp1_ep4_trace_10.json \
  --csv /tmp/moe_fused_tp1_ep4_measured_10.csv \
  --json /tmp/moe_fused_tp1_ep4_measured_10.json
```

Result:

```text
total kernel time = 6413.488 us / 10 = 641.349 us per call

category             total/call   share
moe_grouped_gemm      456.062 us  71.1%
moe_reduce_combine     76.877 us  12.0%
moe_activation         54.400 us   8.5%
memory_misc            43.408 us   6.8%
moe_align_sort         10.602 us   1.7%
```

This matches the measured baseline (`642.048 us`) closely, so the profiler
breakdown is representative of the standalone collector measurement.

The profiler also showed this vLLM warning:

```text
Using default MoE config. Performance might be sub-optimal!
Config file not found ... fused_moe/configs/E=32,N=768,device_name=NVIDIA_GeForce_RTX_4090.json
```

This matters because the case has `E_local=32` and `inter_size=768`. vLLM is
using a default MoE config instead of a tuned RTX 4090 config for this shape.

## What Each Measured Non-GEMM Term Does

### `moe_align_sort`

Time:

```text
10.602 us, 1.7%
```

This prepares token-expert work for grouped GEMM:

```text
count tokens per expert
sort/group token ids by expert
align/pad groups to block size
generate metadata used by grouped GEMM
```

It is not Tensor Core work. It is mostly integer/index processing plus memory
access on CUDA cores.

Shape drivers:

```text
tokens_per_device = num_tokens * topk / ep
local_experts = num_experts / ep
block alignment / padding
routing skew
```

First-pass model:

```text
t_align_sort =
  alpha_align
  + beta_align * tokens_per_device
  + gamma_align * local_experts
```

### `memory_misc`

Time:

```text
43.408 us, 6.8%
```

The sampled kernel was a PyTorch `FillFunctor`, likely clearing or initializing
output/workspace buffers. It is memory-bound CUDA kernel work, not Tensor Core
work.

First-pass model:

```text
t_memory_misc =
  alpha_mem
  + workspace_or_output_bytes / effective_mem_bandwidth
```

For calibration, it may be easier to fit this as a small shape-bucket term until
we know the exact buffers being filled.

### `moe_activation`

Time:

```text
54.400 us, 8.5%
```

The sampled kernel was `vllm::act_and_mul_kernel`, doing SwiGLU:

```text
silu(gate) * up
```

Approximate shape:

```text
elements = tokens_per_device * inter_size
         = 4096 * 768
```

This is CUDA core / SFU / vectorized elementwise work, not Tensor Core work.

First-pass model:

```text
elements = tokens_per_device * inter_size
bytes ~= elements * 3 * dtype_bytes
       # read gate, read up, write activated
flops ~= elements * 5

t_activation =
  max(flops / vector_peak, bytes / mem_bandwidth) * activation_factor
  + alpha_activation
```

### `moe_reduce_combine`

Time:

```text
76.877 us, 12.0%
```

The sampled kernel was a PyTorch `reduce_kernel`. It combines the `topk` expert
outputs back into one output per token, applying routing weights:

```text
out[token] = sum_i topk_weight[token, i] * expert_out[token, i]
```

This is CUDA core + memory hierarchy reduction work, not Tensor Core work.

First-pass model:

```text
elements = num_tokens * topk * hidden
bytes ~= num_tokens * topk * hidden * dtype_bytes
       + num_tokens * topk * 4
       + num_tokens * hidden * dtype_bytes
flops ~= num_tokens * topk * hidden * 2

t_reduce_combine =
  max(flops / vector_peak, bytes / mem_bandwidth) * reduce_factor
  + alpha_reduce
```

## CUDA Core vs Tensor Core

Measured categories:

```text
Tensor Core / GEMM-like:
  moe_grouped_gemm

CUDA core / memory / reduction:
  moe_align_sort
  memory_misc
  moe_activation
  moe_reduce_combine
```

Therefore the MoE model should not apply dense GEMM Tensor Core assumptions to
all fused MoE time. Only the grouped GEMM part should use Tensor Core roofline.
The rest should use vector/core roofline, memory bandwidth, or calibrated
shape-bucket terms.

## Why Dense GEMM Bandwidth Efficiency Does Not Transfer

Dense GEMM on RTX 4090 can reach high memory bandwidth utilization, around 90%
in some measured cases. The MoE grouped GEMM here is different:

```text
many small expert GEMMs
dynamic token-expert grouping
smaller M per expert, about 128 rows/expert
less regular weight access
padding and grouped scheduling overhead
default vLLM MoE config for this E=32,N=768 shape
```

For this case:

```text
roofline memory lower bound = 332.881 us
measured grouped_gemm       = 456.062 us

moe_grouped_gemm_mem_efficiency ~= 332.881 / 456.062 = 0.73
```

If we incorrectly use total measured time to fit grouped GEMM, we get:

```text
332.881 / 642.048 = 0.52
```

That is too conservative because it mixes non-GEMM kernels into the grouped GEMM
efficiency. The better split is:

```text
grouped GEMM efficiency: about 0.70-0.75 for this bucket
non-GEMM overhead:       about 185.3 us for this bucket
```

## Recommended Model Structure

Use a MoE-specific model:

```text
t_moe =
  t_grouped_gemm
  + t_align_sort
  + t_activation
  + t_reduce_combine
  + t_memory_misc
```

With:

```text
t_grouped_gemm =
  max(
    t_compute / moe_grouped_gemm_compute_efficiency,
    t_memory / moe_grouped_gemm_mem_efficiency
  )
```

For the current case:

```text
moe_grouped_gemm_mem_efficiency = 0.73
t_align_sort     = 10.6 us
t_activation     = 54.4 us
t_reduce_combine = 76.9 us
t_memory_misc    = 43.4 us
```

Then:

```text
332.881 / 0.73 + 10.6 + 54.4 + 76.9 + 43.4
~= 641 us
```

This matches measured time.

## Suggested Calibration Buckets

Do not make `0.73` a universal constant. Bucket it by:

```text
device
dtype
framework/version
kernel_source
execution_mode
local_experts = num_experts / ep
inter_size
tokens_per_device = num_tokens * topk / ep
avg_rows_per_expert = tokens_per_device / local_experts
routing_distribution / skew
```

Initial profile-table row:

```text
RTX4090 / bf16 / vllm-0.19.1 / vllm_fused_moe / cudagraph
local_experts = 32
inter_size = 768
tokens_per_device = 4096
avg_rows_per_expert = 128

moe_grouped_gemm_mem_efficiency = 0.73
align_sort_us = 10.6
activation_us = 54.4
reduce_combine_us = 76.9
memory_misc_us = 43.4
```

## Added Scripts

Roofline-vs-measured diagnostic:

```text
scripts/report_moe_internal_breakdown.py
tests/scripts/test_report_moe_internal_breakdown.py
```

Measured CUDA kernel breakdown:

```text
scripts/profile_moe_fused_measured.py
tests/scripts/test_profile_moe_fused_measured.py
```

Validation run:

```text
pytest -q tests/scripts/test_profile_moe_fused_measured.py \
          tests/scripts/test_report_moe_internal_breakdown.py
```

Result:

```text
3 passed
```
