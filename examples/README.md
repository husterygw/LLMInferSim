# Examples

启动真实 vLLM 进程跑某个具体模型, 用于 op dump / 阶段 spike 验证 / debug。
**这些脚本不属于自动化测试**, 不被 `pytest` 收集。若某个 example 对 release 关键,
应在 `tests/e2e/` 写一个薄 wrapper(`e2e` marker, opt-in `RUN_E2E=1`)而不是直接让
pytest 收 examples。测试分层与默认 gate 命令见 `tests/README.md`。

按用途分子目录:

```text
vllm_virtual/  run_platform_selected / run_opt125m / run_qwen3_4b / run_prefix_caching(VirtualPlatform 主验证)
offline/       各模型/并行配置 offline 跑(run_deepseek_* / run_qwen3_*)
serving/       bench_serve_*.sh
pd_disagg/     run_pd_disagg_loopback.py/.sh(PD 分离)
```

## 公共环境变量

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0   # 必加: ifwa env 的 torch_ifwa autoload 会破 torch
export VLLM_VIRTUAL_BACKEND=1            # 激活我们的 OOT VirtualPlatform plugin
export VLLM_USE_V1=1                     # vLLM v1 引擎
export HF_HUB_OFFLINE=1                  # 不联网, 本地 HF cache
export TRANSFORMERS_OFFLINE=1
```

可选:
```bash
export VLLM_LOGGING_LEVEL=DEBUG          # vLLM 内部 debug log
export LLM_INFER_SIM_DUMP_OPS=1           # 0/1/2 — 0 不打 / 1 仅首步 / 2 每步都打
export LLM_INFER_SIM_HW=H100              # 硬件 profile (默认 H100)
export LLM_INFER_SIM_TIME_MODE=realtime   # realtime / instant — 是否真 sleep 估算 latency
```

## 脚本

### `run_platform_selected.py`
最轻量探针: 验证 `vllm.platform_plugins` entry_point 已注册 + plugin function 返
回正确 qualname + `current_platform` 选中 VirtualPlatform。**不启动 LLM**,
~5 秒内跑完。

```bash
VLLM_VIRTUAL_BACKEND=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python examples/vllm_virtual/run_platform_selected.py
```

### `run_opt125m.py`
阶段 0 / 阶段 1 主验证脚本: 启动 vLLM 跑 `facebook/opt-125m`, 3 个不等长 prompt
× max_tokens=3。验证:
- VirtualWorker / VirtualModelRunner 路径走通
- phase 分类 (prefill / decode / mixed) 正确
- cost model 输出非零 latency + breakdown 三栏

```bash
VLLM_VIRTUAL_BACKEND=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_USE_V1=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
LLM_INFER_SIM_DUMP_OPS=1 \
python examples/vllm_virtual/run_opt125m.py
```

### `run_qwen3_4b.py`
阶段 2 主验证脚本: 启动 vLLM 跑 Qwen3-4B-Instruct-2507。验证:
- ProfileManager 解析 Qwen3 hf_config (L=36, hidden=2560, GQA 32/8, head_dim=128)
- llm-viewer dense_layer_time 输出 SwiGLU 三个 GEMM (gate / up / down)
- GQA: q_proj 的 flops ≈ 4 × k_proj / v_proj

需要本地 `/data1/home/ygw268/models/Qwen3-4B-Instruct-2507`, 或通过 `VLLM_INFER_SIM_MODEL`
环境变量指向其他可用路径。

```bash
VLLM_VIRTUAL_BACKEND=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_USE_V1=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
LLM_INFER_SIM_DUMP_OPS=1 \
VLLM_INFER_SIM_MODEL=/data1/home/ygw268/models/Qwen3-4B-Instruct-2507 \
python examples/vllm_virtual/run_qwen3_4b.py 2>&1 | tee /tmp/qwen3_dump.log
```

加 `VLLM_LOGGING_LEVEL=DEBUG` 看 vLLM 内部 debug log。

### `run_prefix_caching.py`
验证 `enable_prefix_caching=True` 命中后 cost model 透明节省 prefill。Batch 1 跑
冷 3500-tok prompt, Batch 2 跑同前缀 + 50 tail; 第二次的 prefill latency 应仅是
第一次的 ~1/13, 自动通过 step_extractor 透传 `num_computed_tokens` 实现。
step log 在命中时会追加 `cached=X computed=Y`。

⚠️ **内存上限假设**: 我们的 `determine_available_memory` 不感知 PrefixCache
block 共享, 真机相同 prefix 的多请求会复用同组 block, 实际容量比我们的估算更大。
当前实现**保守偏差** (cost 不偏快, 吞吐 simulation 偏低), 准确建模需扩 KV
block allocator (与 PD 分离 §10.5 7.6 共享一部分基建)。

```bash
VLLM_VIRTUAL_BACKEND=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_USE_V1=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
LLM_INFER_SIM_TIME_MODE=instant \
python examples/vllm_virtual/run_prefix_caching.py
```

## 与 tests/ 的分工

| 关注 | tests/ | examples/ |
|---|---|---|
| 跑得快 (秒级) | ✓ | 否, 起 vLLM 几十秒 |
| 自动 regression gate | ✓ pytest tests/ | 否, 手动跑 |
| 实例化 LLM | 否, 全 mock | ✓ |
| 用途 | 静态 / 数据结构 / 配置解析 / 接口契约 | op dump 实验 / 模型探索 / debug |
