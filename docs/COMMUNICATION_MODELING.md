# 通信建模改进方案（Phase 5 设计文档）

> 草稿 · 2026-05-16 · 待 review  
> 关联代码：`llm_infer_sim/core/ops/communication.py` + `llm_infer_sim/core/profiles/hardware.py`  
> 实测数据：`/tmp/collective_bench.jsonl`（672 条）/ `/tmp/collective_analysis.csv`  
> 测量工具：`scripts/measure_collectives.py` + `scripts/run_collective_sweep.sh`

---

## 1. 文档目的

在当前 `Plan C` 通信建模（`intra_node_topology` + `protocol_efficiency`）的基础上，根据 RTX 4090 × 8 服务器全维度实测：

```text
6 collective × 5 拓扑配置 × 13 size × 2 mode = 672 数据点
```

提出 Phase 5 的系统性通信建模重构方案。

当前模型的问题不是简单的参数未校准，而是存在结构性偏差：

1. 小消息延迟主要由 fixed / step latency 主导；
2. eager 与 cudagraph 的调用开销完全不同；
3. PCIe 拓扑不能用单一 `1/n` 缩放表达；
4. 不同 collective 的算法形态不同；
5. NCCL 会根据 size / n / topology / hardware 动态选择算法；
6. AllToAll / AllGather / ReduceScatter 的 `data_bytes` 语义必须统一，否则很容易多除或少除一个 `n`。

Phase 5 的目标是把通信模型从：

```text
单一 ring 公式 + 简单 β 缩放
```

升级为：

```text
mode-aware + topology-aware + collective-aware + algorithm-aware
```

并与计算算子的 cost model 统一到同一个抽象下。

---

## 2. 当前模型的真实问题（由实测验证）

| 编号 | 现象 | 根因 | 严重度 |
|---|---|---|---|
| B1 | 小消息 `<1KB` 实测 latency 几乎不随 n 变 | eager 下存在明显 framework dispatch overhead；cudagraph 下主要由 collective step latency 主导 | 高 |
| B2 | 同配置 eager vs cudagraph 差 30-170µs | PyTorch / ATen / ProcessGroupNCCL dispatch overhead 在 cudagraph 下基本消失 | 高 |
| B3 | PCIe 服务器 cross-NUMA 比 same-NUMA 快约 2× | 真瓶颈是 PCIe root 资源；cross-NUMA 让 GPU 分散到不同 root | 高 |
| B4 | Broadcast n=8 大消息偏差 +1500% | 当前模型错误地使用 ring；实测更接近 tree | 高 |
| B5 | 不同 collective 的 overhead 差异明显 | AllGather / ReduceScatter / AllToAll / Broadcast 的实现复杂度和通信模式不同 | 中 |
| B6 | n=8 实测 β 跟 n=4 接近，不符合简单 `1/n` 趋势 | 4090 server 是双 PCIe root，n>4 后通信分布发生变化 | 高 |
| B7 | NCCL 算法选择是动态的 | NCCL 根据 size / n / topology / hardware 选择 ring / tree / NVLS / direct 等算法 | 高 |

### 2.1 AllReduce eager vs cudagraph（同 NUMA n=2）

| size | eager µs | cudagraph µs | delta |
|---|---:|---:|---:|
| 1 KB | 147 | 12 | 135 |
| 64 KB | 70 | 22 | 48 |
| 4 MB | 597 | 568 | 29 |
| 256 MB | 35024 | 35173 | 噪声主导 |

关键观察：

```text
小消息下，framework overhead 贡献 80-90% latency；
大消息下，framework overhead 被带宽项摊薄到 <1%。
```

### 2.2 Cross-NUMA vs Same-NUMA（AllReduce 4MB）

| n | same_numa µs | cross_numa µs | ratio |
|---|---:|---:|---:|
| 2 | 604 | 380 | 0.63 |
| 4 | 1339 | 717 | 0.54 |

关键观察：

```text
cross-NUMA 反而更快，因为 GPU 分散到不同 PCIe root，
每个 root 的带宽压力更低。
```

### 2.3 Broadcast 公式 vs 实测（n=8，大消息）

| size | 实测 | ring 预测 | tree 预测 |
|---|---:|---:|---:|
| 4 MB | 698 µs | 11828 µs | 1009 µs |
| 67 MB | 11.8 ms | 188 ms | 16.8 ms |
| 256 MB | 47.0 ms | 752 ms | 67.2 ms |

关键观察：

```text
Broadcast 大消息更接近 tree，而不是 ring。
当前 ring 建模会造成 14-16× 偏差。
```

---

## 3. 设计哲学：3 层分解 + 算法选择层

### 3.1 与计算算子对齐

计算算子当前可抽象为：

```text
T_compute = math_term + memory_term + dispatch_overhead × [mode == eager]
```

通信算子也应当使用对称结构：

```text
T_collective = algorithm_term + framework_call_overhead × [mode == eager] + cross_node_term
```

其中：

```text
algorithm_term:
  NCCL collective 的主体通信开销，包括 step latency 和数据传输项。

framework_call_overhead:
  PyTorch / ATen / ProcessGroupNCCL / CUDA launch 等 eager-only 调用开销。

cross_node_term:
  跨节点通信项，Phase 5 暂时保留 hook，不做大改。
```

### 3.2 Phase 5 默认不引入独立 `nccl_kernel_floor`

上一版曾考虑：

```text
T = nccl_kernel_floor + framework_overhead + algorithm_term
```

Phase 5 最终改为：

```text
T = algorithm_term + framework_overhead × [mode == eager]
```

原因是当前 RTX 4090 实测显示，cudagraph 小消息下的最小延迟可以被 `algorithm_term(data→0)` 较好解释。

例如，设 PCIe `comm_step_latency = 5µs`：

| 配置 | 公式 data→0 | 实测 1KB cudagraph |
|---|---:|---:|
| AllReduce n=2 ring | `2 × 1 × 5 = 10µs` | 12µs |
| Broadcast n=8 tree | `ceil(log2(8)) × 5 = 15µs` | 17µs |
| AllGather n=4 ring | `3 × 5 = 15µs` | 13µs |

因此，Phase 5 默认认为：

```text
cudagraph 小消息 floor ≈ algorithm_term(data→0)
```

但注意：这是当前 RTX 4090 PCIe 机器上的经验结论，不应写死为所有硬件上的真理。

为后续 H100 / B200 / NVSwitch / 跨节点场景保留扩展点：

```python
optional_collective_floor: dict[str, float] | None = None
```

默认不启用。若未来发现 NVLink / NVLS / inter-node 小消息系统性低估，再开启该字段。

### 3.3 `comm_step_latency` 的语义

原字段名 `link_latency` 容易误导，因为它不是裸 PCIe / NVLink 物理链路延迟。

Phase 5 建议改名为：

```python
comm_step_latency
```

语义为：

```text
NCCL collective 每个逻辑通信 step 的 effective latency。
```

它包含：

```text
链路访问延迟
GPU 间路由延迟
NCCL step 内部同步开销
小包协议开销的一部分
```

因此它不是纯硬件 SerDes latency，而是 cost model 中用于 collective 公式的有效 step latency。

推荐默认值：

```text
NVLink / NVSwitch: 1µs
PCIe 4.0 / 5.0: 5µs
```

---

## 4. 统一数据语义：`data_bytes = per-rank input bytes`

Phase 5 必须明确所有 collective 函数的 `data_bytes` 语义：

```text
data_bytes 永远表示每个 rank 本次 collective 的 input bytes。
```

这点非常关键。否则 AllGather / ReduceScatter / AllToAll 很容易多除或少除一个 `n`。

### 4.1 各 collective 的 data 语义

| collective | `data_bytes` 语义 |
|---|---|
| AllReduce | 每个 rank 上待 reduce 的 tensor bytes |
| Broadcast | root rank 要广播的 tensor bytes |
| AllGather | 每个 rank 输入 shard bytes |
| ReduceScatter | 每个 rank 输入 full tensor bytes |
| AllToAll | 每个 rank 总输入 bytes，即要发给所有 peer 的总量 |
| P2P | send bytes |

### 4.2 对应 ring / pairwise 数据项

在 `data_bytes = per-rank input bytes` 约定下：

| collective | 数据通信因子 |
|---|---:|
| AllReduce ring | `2(n-1)/n × data_bytes` |
| Broadcast tree | `ceil(log2(n)) × data_bytes` |
| Broadcast ring | `(n-1) × data_bytes` |
| AllGather ring | `(n-1) × data_bytes` |
| ReduceScatter ring | `(n-1)/n × data_bytes` |
| AllToAll pairwise | `(n-1)/n × data_bytes` |
| P2P | `data_bytes` |

---

## 5. `HardwareConfig` 新增 / 修改字段

```python
@dataclass
class HardwareConfig:
    # === 原有拓扑字段 ===
    intra_node_bandwidth: float = 450e9
    intra_node_size: int = 8
    intra_node_topology: str = "nvlink_full"      # "nvlink_full" | "pcie_shared_root"
    intra_node_protocol_efficiency: float = 0.7
    comm_efficiency: float = 1.0

    # === PCIe 拓扑字段 ===
    intra_node_gpus_per_root: int = 8
    intra_node_num_roots: int = 1

    # 可选：显式描述 GPU -> root 映射。
    # 如果没有，则 fallback 到 gpu_id // gpus_per_root。
    gpu_to_root: dict[int, int] | None = None

    # === NCCL / NVSwitch 能力 ===
    has_nvlink_sharp: bool = False

    # 默认不开启 NVLS 公式。避免未校准时 H100/B200 过度乐观。
    enable_nvls_model: bool = False

    # === 通信 step latency ===
    # 注意：不是裸物理链路 latency，而是 collective step 的 effective latency。
    # PCIe profile 可设为 5e-6；NVLink/NVSwitch 可设为 1e-6。
    comm_step_latency: float = 1e-6

    # === 可选扩展：collective 小消息 floor ===
    # 默认 None，不参与计算。
    # 若未来发现某些硬件上 algorithm_term(data->0) 系统性低估，
    # 可用该字段补偿。
    optional_collective_floor: dict[str, float] | None = None

    # === 可选扩展：framework overhead override ===
    # 默认 None，使用 communication.py 里的 DEFAULT_FRAMEWORK_CALL_OVERHEAD。
    framework_call_overhead: dict[str, float] | None = None

    # === 可选扩展：算法 bias ===
    # 用于修正 NCCL 实际算法选择和简化公式之间的差异。
    # 默认 None，所有 bias = 1.0。
    collective_algo_bias: dict[str, dict[str, float]] | None = None
```

---

## 6. `effective_intra_bw` 拓扑感知

```python
TopologyHint = Literal["concentrated", "balanced"]

def effective_intra_bw(
    self,
    n: int = 1,
    topology_hint: TopologyHint = "concentrated",
    visible_devices: list[int] | None = None,
) -> float:
    """
    返回 collective 公式中的有效单向 β，单位 B/s。
    """

    raw = (
        self.intra_node_bandwidth / 2
        * self.comm_efficiency
        * self.intra_node_protocol_efficiency
    )

    if self.intra_node_topology == "nvlink_full":
        return raw

    if self.intra_node_topology == "pcie_shared_root":
        n_per_root = self._estimate_n_per_root(
            n=n,
            topology_hint=topology_hint,
            visible_devices=visible_devices,
        )
        return raw / max(n_per_root, 1)

    return raw
```

辅助函数：

```python
def _estimate_n_per_root(
    self,
    n: int,
    topology_hint: TopologyHint,
    visible_devices: list[int] | None = None,
) -> int:
    if visible_devices:
        roots = []
        for gpu_id in visible_devices[:n]:
            if self.gpu_to_root is not None:
                root = self.gpu_to_root.get(gpu_id)
                if root is None:
                    root = gpu_id // max(self.intra_node_gpus_per_root, 1)
            else:
                root = gpu_id // max(self.intra_node_gpus_per_root, 1)
            roots.append(root)

        counts = Counter(roots)
        return max(counts.values()) if counts else 1

    if topology_hint == "balanced":
        return math.ceil(n / max(self.intra_node_num_roots, 1))

    return min(n, self.intra_node_gpus_per_root)
```

### 6.1 RTX 4090 server 自洽 case

假设：

```text
PCIe 4.0 ×16 bidir nominal = 64GB/s
unidir = 32GB/s
protocol_eff = 0.7
raw β = 22.4GB/s
num_roots = 2
gpus_per_root = 4
```

| n | topology_hint | n_per_root | β_eff |
|---|---|---:|---:|
| 1 | 任意 | 1 | 22.4 GB/s |
| 2 | concentrated | 2 | 11.2 GB/s |
| 2 | balanced | 1 | 22.4 GB/s |
| 4 | concentrated | 4 | 5.6 GB/s |
| 4 | balanced | 2 | 11.2 GB/s |
| 8 | balanced | 4 | 5.6 GB/s |

这与实测趋势一致：

```text
same_numa_n=4 β ≈ 5GB/s
cross_numa_n=4 β ≈ 10GB/s
```

---

## 7. 默认 framework overhead 与 override

模块级默认值：

```python
DEFAULT_FRAMEWORK_CALL_OVERHEAD: dict[str, float] = {
    "allreduce":     50e-6,
    "allgather":     85e-6,
    "reducescatter": 100e-6,
    "alltoall":      40e-6,
    "broadcast":     30e-6,
    "p2p":           100e-6,
}
```

获取方式：

```python
def _framework_overhead(
    hw: HardwareConfig,
    collective: str,
    mode: CollectiveMode,
) -> float:
    if mode == "cudagraph":
        return 0.0

    table = hw.framework_call_overhead or DEFAULT_FRAMEWORK_CALL_OVERHEAD
    return table.get(collective, table.get("default", 0.0))
```

说明：

```text
framework_call_overhead 不是纯硬件参数；
它主要与 PyTorch / NCCL / CUDA runtime / ProcessGroupNCCL 调用路径有关。
```

但在 simulator 中，它仍然适合允许 profile override，因为不同机器上的软件环境通常和 hardware profile 一起校准。

---

## 8. 算法库：per-collective 候选算法

### 8.1 AllReduce

```python
def _allreduce_ring(n: int, data: float, alpha: float, beta: float) -> float:
    # data = per-rank input bytes
    return 2 * (n - 1) * alpha + (2 * (n - 1) / n) * data / beta


def _allreduce_tree(n: int, data: float, alpha: float, beta: float) -> float:
    # reduce up + broadcast down
    # 保守模型：每层传 full data
    depth = math.ceil(math.log2(n))
    return 2 * depth * alpha + 2 * data / beta


def _allreduce_nvls(n: int, data: float, alpha: float, beta_aggregate: float) -> float:
    # Experimental:
    # NVSwitch SHARP / NVLS 的真实行为复杂，这里只作为可选候选。
    return 2 * alpha + data / beta_aggregate
```

注意：

```text
_allreduce_tree 的 data 项不建议写成 2 * log2(n) * data / beta。
否则会把 tree 大消息估得过慢。

这里采用：
  latency 项随 log2(n) 增长；
  data 项用经验化的 2 × data / beta 表示 reduce + broadcast。
```

### 8.2 Broadcast

```python
def _broadcast_ring(n: int, data: float, alpha: float, beta: float) -> float:
    # data = root broadcast bytes
    return (n - 1) * alpha + (n - 1) * data / beta


def _broadcast_tree(n: int, data: float, alpha: float, beta: float) -> float:
    depth = math.ceil(math.log2(n))
    return depth * alpha + depth * data / beta
```

### 8.3 AllGather

```python
def _allgather_ring(n: int, data: float, alpha: float, beta: float) -> float:
    # data = per-rank input shard bytes
    # 每轮发送一个 shard，共 n-1 轮
    return (n - 1) * alpha + (n - 1) * data / beta
```

### 8.4 ReduceScatter

```python
def _reducescatter_ring(n: int, data: float, alpha: float, beta: float) -> float:
    # data = per-rank input full tensor bytes
    # 每个 rank 最终得到 data/n，ring reduce-scatter 总传输因子 (n-1)/n
    return (n - 1) * alpha + ((n - 1) / n) * data / beta
```

### 8.5 AllToAll

```python
def _alltoall_pairwise(n: int, data: float, alpha: float, beta: float) -> float:
    # data = per-rank total input bytes
    # 每轮和一个 peer 交换 data/n，共 n-1 轮
    return (n - 1) * alpha + ((n - 1) / n) * data / beta
```

注意：不要写成：

```python
data / (n * n * beta)
```

除非 `data` 表示全局所有 rank 的总数据量。Phase 5 统一采用：

```text
data = per-rank input bytes
```

因此 AllToAll 的数据项应为：

```text
(n-1)/n × data / beta
```

### 8.6 P2P

```python
def _p2p_single(data: float, alpha: float, beta: float) -> float:
    return alpha + data / beta
```

---

## 9. 算法选择：`min(candidate × algo_bias)`

NCCL 实际会根据：

```text
collective type
message size
rank 数
GPU 拓扑
GPU 架构
NCCL 版本
network
```

动态选择 algorithm 和 protocol。

Phase 5 不复刻 NCCL 内部阈值，而采用近似：

```text
selected_algo = argmin(candidate_time × algo_bias)
```

### 9.1 bias 获取方式

```python
def _algo_bias(hw: HardwareConfig, collective: str, algo: str) -> float:
    if hw.collective_algo_bias is None:
        return 1.0

    return (
        hw.collective_algo_bias
        .get(collective, {})
        .get(algo, 1.0)
    )
```

### 9.2 选择函数

```python
def _select_min_candidate(
    hw: HardwareConfig,
    collective: str,
    candidates: dict[str, float],
) -> tuple[str, float]:
    best_algo = None
    best_time = float("inf")

    for algo, t in candidates.items():
        adjusted = t * _algo_bias(hw, collective, algo)
        if adjusted < best_time:
            best_time = adjusted
            best_algo = algo

    assert best_algo is not None
    return best_algo, best_time
```

### 9.3 为什么需要 algo_bias

`min(candidate)` 的优点：

```text
不需要 hardcode NCCL threshold；
NCCL 改阈值时模型不用频繁跟进；
符合“选择最快算法”的总体目标。
```

但简化公式不一定和 NCCL 完全一致。

例如：

```text
理论上 tree 更快，但 NCCL 因 threshold / tuning table 仍选择 ring；
或者 tree 公式系统性高估 / 低估。
```

因此预留 `algo_bias`：

```python
collective_algo_bias = {
    "broadcast": {
        "tree": 0.75,
        "ring": 1.0,
    },
    "allreduce": {
        "ring": 1.0,
        "tree": 1.2,
        "nvls": 1.0,
    },
}
```

默认全部为 1.0，不影响第一版。

---

## 10. 顶层 API 设计

### 10.1 CollectiveMode

```python
CollectiveMode = Literal["eager", "cudagraph"]
TopologyHint = Literal["concentrated", "balanced"]
```

### 10.2 AllReduce

```python
def allreduce_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: list[int] | None = None,
    cross_node: bool = False,
) -> float:
    """
    AllReduce wall-clock time.

    data_bytes:
        per-rank input tensor bytes.
    """

    if n <= 1 or data_bytes <= 0:
        return 0.0

    if cross_node:
        return _hierarchical_allreduce(
            data_bytes, n, hw,
            mode=mode,
            topology_hint=topology_hint,
            visible_devices=visible_devices,
        )

    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(
        n=n,
        topology_hint=topology_hint,
        visible_devices=visible_devices,
    )

    candidates = {
        "ring": _allreduce_ring(n, data_bytes, alpha, beta),
        "tree": _allreduce_tree(n, data_bytes, alpha, beta),
    }

    if hw.has_nvlink_sharp and hw.enable_nvls_model:
        candidates["nvls"] = _allreduce_nvls(
            n, data_bytes, alpha, beta_aggregate=beta * n
        )

    _, algo_term = _select_min_candidate(hw, "allreduce", candidates)

    optional_floor = _optional_collective_floor(hw, "allreduce")
    fw_oh = _framework_overhead(hw, "allreduce", mode)

    return optional_floor + algo_term + fw_oh
```

### 10.3 Broadcast

```python
def broadcast_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: list[int] | None = None,
) -> float:
    """
    Broadcast wall-clock time.

    data_bytes:
        root rank broadcast tensor bytes.
    """

    if n <= 1 or data_bytes <= 0:
        return 0.0

    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(n, topology_hint, visible_devices)

    candidates = {
        "ring": _broadcast_ring(n, data_bytes, alpha, beta),
        "tree": _broadcast_tree(n, data_bytes, alpha, beta),
    }

    _, algo_term = _select_min_candidate(hw, "broadcast", candidates)

    return (
        _optional_collective_floor(hw, "broadcast")
        + algo_term
        + _framework_overhead(hw, "broadcast", mode)
    )
```

### 10.4 AllGather

```python
def allgather_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: list[int] | None = None,
) -> float:
    """
    AllGather wall-clock time.

    data_bytes:
        per-rank input shard bytes.
    """

    if n <= 1 or data_bytes <= 0:
        return 0.0

    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(n, topology_hint, visible_devices)

    algo_term = _allgather_ring(n, data_bytes, alpha, beta)

    return (
        _optional_collective_floor(hw, "allgather")
        + algo_term
        + _framework_overhead(hw, "allgather", mode)
    )
```

### 10.5 ReduceScatter

```python
def reducescatter_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: list[int] | None = None,
) -> float:
    """
    ReduceScatter wall-clock time.

    data_bytes:
        per-rank input full tensor bytes.
    """

    if n <= 1 or data_bytes <= 0:
        return 0.0

    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(n, topology_hint, visible_devices)

    algo_term = _reducescatter_ring(n, data_bytes, alpha, beta)

    return (
        _optional_collective_floor(hw, "reducescatter")
        + algo_term
        + _framework_overhead(hw, "reducescatter", mode)
    )
```

### 10.6 AllToAll

```python
def alltoall_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: list[int] | None = None,
) -> float:
    """
    AllToAll wall-clock time.

    data_bytes:
        per-rank total input bytes, i.e. total bytes this rank sends to all peers.
    """

    if n <= 1 or data_bytes <= 0:
        return 0.0

    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(n, topology_hint, visible_devices)

    algo_term = _alltoall_pairwise(n, data_bytes, alpha, beta)

    return (
        _optional_collective_floor(hw, "alltoall")
        + algo_term
        + _framework_overhead(hw, "alltoall", mode)
    )
```

### 10.7 P2P

```python
def p2p_time(
    data_bytes: float,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: list[int] | None = None,
) -> float:
    if data_bytes <= 0:
        return 0.0

    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(
        n=2,
        topology_hint=topology_hint,
        visible_devices=visible_devices,
    )

    algo_term = _p2p_single(data_bytes, alpha, beta)

    return (
        _optional_collective_floor(hw, "p2p")
        + algo_term
        + _framework_overhead(hw, "p2p", mode)
    )
```

### 10.8 Optional collective floor

```python
def _optional_collective_floor(hw: HardwareConfig, collective: str) -> float:
    if hw.optional_collective_floor is None:
        return 0.0

    return hw.optional_collective_floor.get(
        collective,
        hw.optional_collective_floor.get("default", 0.0),
    )
```

---

## 11. mode 如何传递

### 11.1 从 vLLM 配置推断

```text
enforce_eager=True
  -> mode = "eager"

production CUDA Graph enabled
  -> mode = "cudagraph"
```

建议在全局 profile 中增加：

```python
execution_mode: Literal["eager", "cudagraph"] = "eager"
```

然后计算算子与通信算子共用同一个字段：

```text
compute op overhead
communication framework overhead
```

都由 `execution_mode` 控制。

### 11.2 计算算子也要 mode-aware

当前 `_get_kernel_overhead` 若无论 eager / cudagraph 都加 `2µs/op`，会导致 cudagraph 下系统性高估。

修复：

```python
def _get_kernel_overhead(self, op_category: str, mode: str = "eager") -> float:
    if mode == "cudagraph":
        return 0.0

    return self.hw.kernel_overhead.get(
        op_category,
        self.hw.kernel_overhead["default"],
    )
```

影响示例：

```text
Qwen3-4B 若有 360 个 op：
360 × 2µs = 0.72ms/step
```

这会明显影响 decode step 估计。

---

## 12. topology_hint 如何传递

推荐优先级：

```text
1. 用户通过环境变量显式指定
2. 从 CUDA_VISIBLE_DEVICES + gpu_to_root 自动推断
3. fallback 到 concentrated
```

### 12.1 环境变量

```bash
export LLM_INFER_SIM_NUMA_HINT=balanced
```

可选值：

```text
concentrated
balanced
```

### 12.2 自动推断

如果用户设置：

```bash
CUDA_VISIBLE_DEVICES=0,4
```

并且 profile 中有：

```python
gpu_to_root = {
    0: 0, 1: 0, 2: 0, 3: 0,
    4: 1, 5: 1, 6: 1, 7: 1,
}
```

则自动判断为：

```text
balanced
```

如果是：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

则判断为：

```text
concentrated
```

若无法推断，默认：

```text
concentrated
```

原因：

```text
保守估计更安全；
sim 偏慢比 sim 偏快更容易解释。
```

---

## 13. Profile 示例：RTX 4090 server

```python
"RTX_4090": dict(
    peak_flops_bf16=165.2e12,
    mem_bandwidth=1008e9,
    onchip_buffer=72 * 1024 * 1024,

    # 拓扑
    intra_node_bandwidth=64e9,               # PCIe 4.0 ×16 bidir nominal
    intra_node_topology="pcie_shared_root",
    intra_node_gpus_per_root=4,
    intra_node_num_roots=2,
    gpu_to_root={
        0: 0, 1: 0, 2: 0, 3: 0,
        4: 1, 5: 1, 6: 1, 7: 1,
    },
    intra_node_protocol_efficiency=0.7,

    # 通信 step latency
    comm_step_latency=5e-6,

    # 硬件能力
    has_nvlink_sharp=False,
    enable_nvls_model=False,

    # 单节点
    inter_node_bandwidth=0.0,

    # 默认不启用 optional floor
    optional_collective_floor=None,

    # 可选 override；默认 None，走模块级默认值
    framework_call_overhead=None,

    # 可选 algo bias；默认 None，全部为 1.0
    collective_algo_bias=None,
)
```

---

## 14. H100 / B200 profile 注意事项

对于 H100 / H200 / B200 SXM：

```python
intra_node_topology = "nvlink_full"
comm_step_latency = 1e-6
has_nvlink_sharp = True
```

但建议：

```python
enable_nvls_model = False
```

直到有 H100 / B200 nccl-tests 实测数据再打开。

原因：

```text
NVLS / NVSwitch SHARP 的有效带宽不是简单 n × β；
直接加入 NVLS 候选可能让 allreduce 过于乐观。
```

启用 NVLS 的建议条件：

```text
1. 有目标机器 nccl-tests all_reduce_perf 数据；
2. H100/B200 profile 已用实测校准过 β 和 algo_bias；
3. 端到端 TP allreduce 误差需要 NVLS 才能解释。
```

---

## 15. Protocol 层暂不显式建模

NCCL 还有：

```text
LL
LL128
Simple
```

三种 protocol。它们会影响：

```text
小消息 latency
中等消息带宽爬升
大消息 bandwidth plateau
```

Phase 5 暂不显式建模 protocol，原因：

```text
1. 显式 protocol 会使组合数从 algorithm 扩展到 algorithm × protocol；
2. 当前 672 条数据暴露的最大结构性误差主要来自 mode / topology / algorithm；
3. 可以先通过 comm_step_latency、β_eff、framework_oh、algo_bias 间接吸收 protocol 影响。
```

但需要承认：

```text
decode 小 batch / 小 TP allreduce 下，protocol 可能影响明显。
```

如果 Phase 5 后小消息误差仍然较大，再进入 Phase 6：

```text
algorithm × protocol selection
size-dependent effective_bw
latency S-curve fitting
```

---

## 16. 跨节点 inter-node 处理

Phase 5 暂不重构跨节点，只保留 hook。

```python
def _hierarchical_allreduce(
    data: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode,
    topology_hint: TopologyHint,
    visible_devices: list[int] | None = None,
) -> float:
    n1 = hw.intra_node_size
    n2 = math.ceil(n / n1)

    # Phase 5:
    # intra 部分调用新的 allreduce_time / effective_intra_bw
    # inter 部分沿用现有 inter_node_bandwidth / inter_node_latency 公式
    ...
```

Backlog：

```text
等跨节点机器可用后，使用 measure_collectives.py 跑 inter-node sweep，
再校准 hierarchical allreduce / allgather / alltoall。
```

---

## 17. 测试与验证计划

### 17.1 单元测试

`tests/core/ops/test_communication.py`

覆盖：

```text
1. AllReduce ring / tree 公式；
2. Broadcast ring / tree 公式；
3. AllGather data_bytes = per-rank input shard；
4. ReduceScatter data_bytes = per-rank input full tensor；
5. AllToAll data_bytes = per-rank total input；
6. eager vs cudagraph 差值 = framework_oh；
7. optional_collective_floor 默认不生效；
8. algo_bias 默认 1.0；
9. topology_hint concentrated vs balanced；
10. nvlink_full 下 effective_intra_bw 不随 n 缩放。
```

### 17.2 data semantics 测试

必须加专门测试，避免未来误改：

```python
def test_allgather_data_semantics():
    # n=4, data=1MB per rank input shard
    # ring data factor should be 3MB, not 0.75MB
    ...

def test_alltoall_data_semantics():
    # n=4, data=4MB per rank total input
    # each peer gets 1MB, 3 rounds -> 3MB total factor
    ...
```

### 17.3 集成测试

`tests/core/test_inter_node_cost_consistency.py`

要求：

```text
1. 原 hierarchical 测试不挂；
2. mode 参数透传正确；
3. topology_hint 参数透传正确。
```

### 17.4 数据回归

实施后重跑：

```bash
bash scripts/run_collective_sweep.sh
python scripts/analyze_collectives.py
```

目标：

```text
每个 (collective, mode, label) 平均 gap < 30%
中位 gap < 20%
Broadcast 大消息 gap 从 +1500% 收敛到 < ±30%
AllReduce eager 大消息 cross-NUMA hint=balanced 后 gap < 20%
```

### 17.5 端到端验证

```text
1. Qwen3-32B TP=4 same NUMA: GPU 0-3
   topology_hint=concentrated
   目标 TTFT/TPOT ±15%

2. Qwen3-4B TP=2 same NUMA: GPU 0,1
   topology_hint=concentrated
   目标 ±20%

3. Qwen3-4B TP=2 cross NUMA: GPU 0,4
   topology_hint=balanced
   目标 ±15%

4. Qwen3-32B TP=8 full node
   topology_hint=balanced
   目标 ±15%
```

---

## 18. 实施计划

| 阶段 | 工作 | 文件 | 预估 |
|---|---|---|---:|
| 5a | 加 `gpus_per_root / num_roots / gpu_to_root / has_nvlink_sharp / enable_nvls_model / comm_step_latency` 等字段 | `hardware.py` | 40 min |
| 5b | `effective_intra_bw` 改为 method，并支持 `topology_hint / visible_devices` | `hardware.py` | 40 min |
| 5c | 明确 `data_bytes = per-rank input bytes`，重写各 collective 公式 | `communication.py` | 1 h |
| 5d | 加 `mode / topology_hint / visible_devices` 参数 | `communication.py` | 40 min |
| 5e | 加 `framework overhead override / optional floor / algo_bias` | `communication.py` | 40 min |
| 5f | 新增 `broadcast_time / reducescatter_time` | `communication.py` | 30 min |
| 5g | 计算 op 修复 mode-aware `kernel_overhead` | `roofline.py` | 20 min |
| 5h | 上游 cost model 调用点透传 mode/topology_hint | 多文件 | 40 min |
| 5i | RTX 4090 profile 更新；A100/H100/B200 profile 加字段 | `hardware.py` | 40 min |
| 5j | 单元测试与回归测试 | `tests/` | 1.5 h |
| 5k | 跑 collective sweep + bench_compare | `scripts/` | 1.5 h |
| 5l | 文档更新 | `docs/` | 30 min |

总计：

```text
约 8-9 小时
```

相比原计划 6-7 小时略有增加，主要因为补充了：

```text
data semantics 测试
visible_devices 推断
framework override / algo_bias 扩展点
compute mode-aware 修复
```

---

## 19. 风险与 unknown

| 风险 | 说明 | 缓解 |
|---|---|---|
| `algorithm_term(data→0)` 不一定能解释所有硬件小包 floor | RTX 4090 上成立，但 H100 / NVSwitch / inter-node 未验证 | 保留 `optional_collective_floor` |
| `comm_step_latency` 不是裸物理 latency | 它是 collective step effective latency | 字段命名和注释明确 |
| `min(candidate)` 不等于 NCCL 真实选择 | NCCL 使用 tuning table / heuristic | 预留 `algo_bias` |
| AllGather / ReduceScatter / AllToAll data 口径易错 | 不同 benchmark 工具口径不同 | 统一 `per-rank input bytes` 并加单测 |
| NVLS 公式过于粗糙 | NVSwitch SHARP 行为复杂 | 默认 `enable_nvls_model=False` |
| framework overhead 与软件栈相关 | PyTorch/NCCL/CUDA 版本会影响 | profile 支持 override |
| PCIe root 映射不一定连续 | GPU 编号不一定等于拓扑编号 | 支持 `gpu_to_root` |
| protocol 层未建模 | 小消息 decode 可能受影响 | Phase 6 加 size-dependent / protocol 模型 |

---

## 20. 还需要决定的问题

### 20.1 `topology_hint` 默认 concentrated 还是 balanced？

推荐：

```text
默认 concentrated
```

理由：

```text
保守估计更安全；
sim 偏慢比 sim 偏快更容易解释；
如果用户明确知道跨 NUMA，可设 balanced。
```

### 20.2 `mode` 默认 eager 还是 cudagraph？

推荐：

```text
不要在底层模块硬编码默认；
从 BackendProfile / vLLM config 自动推断。
```

兼容旧 API 可暂设：

```python
mode="eager"
```

但实际上游必须传入真实 execution mode。

### 20.3 framework overhead 放模块级还是 profile？

推荐：

```text
模块级默认值 + profile override
```

即：

```python
DEFAULT_FRAMEWORK_CALL_OVERHEAD
hw.framework_call_overhead
```

### 20.4 是否同时改跨节点？

推荐：

```text
Phase 5 不改跨节点主体公式，只留 hook；
等 inter-node 实测后再重构。
```

### 20.5 是否复刻 NCCL threshold？

推荐：

```text
不复刻；
使用 min(candidate × algo_bias)。
```

理由：

```text
NCCL threshold 版本相关；
复刻成本高；
实测校准表更靠谱。
```

### 20.6 是否显式建模 protocol？

推荐：

```text
Phase 5 不建；
Phase 6 若小消息误差仍大，再加。
```

---

## 21. 跟现有代码的兼容性

### 21.1 Breaking changes

```text
1. hw.effective_intra_bw 从 property 变 method；
2. communication 函数增加 keyword args；
3. link_latency 建议改名为 comm_step_latency；
4. AllGather / AllToAll 的 data_bytes 语义必须统一。
```

### 21.2 旧调用兼容

为避免大量 break，可保留默认参数：

```python
mode: CollectiveMode = "eager"
topology_hint: TopologyHint = "concentrated"
visible_devices: list[int] | None = None
```

但上游 cost model 应逐步显式传入。

### 21.3 grep 调用点

```bash
grep -rn "allreduce_time\|allgather_time\|alltoall_time\|p2p_time"   llm_infer_sim --include="*.py"
```

需要检查每个调用点的：

```text
data_bytes 口径
mode
topology_hint
visible_devices
```

---

## 22. 文件清单

| 文件 | 状态 | 改动 |
|---|---|---|
| `llm_infer_sim/core/profiles/hardware.py` | 修改 | 新字段 + `effective_intra_bw` method |
| `llm_infer_sim/core/ops/communication.py` | 大改 | 算法库 + 3 层公式 + 6 个 collective |
| `llm_infer_sim/core/cost_model/*.py` | 修改 | 透传 mode / topology_hint / visible_devices |
| `llm_infer_sim/core/roofline.py` | 修改 | `kernel_overhead` mode-aware |
| `tests/core/ops/test_communication.py` | 新增或扩展 | 公式、mode、topology、data semantics 测试 |
| `tests/core/test_inter_node_cost_consistency.py` | 修改 | mode 参数与兼容性 |
| `tests/core/test_hardware_profiles.py` | 修改 | 新字段断言 |
| `docs/COMMUNICATION_MODELING.md` | 更新 | 本文档 |
| `docs/PROJECT_REPORT.md` | 更新 | 通信建模章节 |
| `scripts/README.md` | 更新 | measure_collectives 使用说明 |

---

## 23. 一句话总结

Phase 5 的核心不是继续调 `α/β`，而是修正通信建模的结构：

```text
T_collective =
    optional_collective_floor
  + selected_algorithm_term(comm_step_latency, effective_bw, n, data)
  + framework_call_overhead × [mode == eager]
  + cross_node_term
```

其中：

```text
data_bytes 统一为 per-rank input bytes；
effective_bw 根据 topology_hint / visible_devices 感知 PCIe root；
selected_algorithm 使用 min(candidate × algo_bias)；
framework overhead 支持 profile override；
optional floor 默认关闭，作为 H100/B200/跨节点 fallback；
计算 op 的 kernel_overhead 也同步改成 mode-aware。
```

这套方案的好处是：

```text
1. 解释 RTX 4090 实测暴露出的结构性偏差；
2. 保持与计算算子 cost model 对称；
3. 避免为单台机器过拟合；
4. 给 H100/B200/NVSwitch/跨节点保留扩展点；
5. 为后续 vLLM TP / EP / MoE 通信建模打基础。
```
