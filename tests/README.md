# tests/

模拟器核心的自动化测试。默认 **CPU-only、不启动真实 vLLM LLM、不依赖本地模型或
真实 collector 数据**。collector 采集管线有独立的 `collector/tests/`,顶层
`pyproject.toml` 把两者都纳入默认 `pytest`。

## 默认 PR gate

```bash
conda activate llm_sim
pytest tests collector/tests \
  -m "not gpu and not e2e and not realdata and not slow and not nightly" -q
```

裸 `pytest` 也是干净的:根 `conftest.py` 的 `pytest_collection_modifyitems`
会自动 skip 环境敏感测试,不用每次记那条长 `-m` 命令:

- `gpu` — 无 CUDA 时自动 skip。
- `e2e` / `realdata` / `nightly` — 未显式 opt-in(对应环境变量 = `1`)时 skip。
- `slow` 与环境无关,不自动 skip;PR gate 命令里显式 `-m "not slow"` 排除。

## marker

定义在 `pyproject.toml` 的 `[tool.pytest.ini_options].markers`。

| marker | 含义 | 默认进 PR gate |
|---|---|---|
| `unit` | 纯 CPU 单元(单函数 / dataclass / 纯计算) | 是 |
| `contract` | API / schema / signature / operator 静态契约 | 是 |
| `integration` | 多模块 CPU 组合,mock / synthetic data | 是 |
| `tools` | `scripts/` 的 CLI / report / case 生成器行为 | 是(须 fast、CPU-only、临时目录自给) |
| `e2e` | 启动真实 runtime 或跑文档化示例 | 否(opt-in:`RUN_E2E=1`) |
| `gpu` | 需要 CUDA / GPU | 否(无 CUDA 自动 skip) |
| `realdata` | 需要本地 collector 数据资产 | 否(opt-in:`RUN_REALDATA=1`) |
| `slow` | 预计数秒以上 | 否(gate 命令显式排除) |
| `nightly` | 不适合 PR gate | 否(opt-in:`RUN_NIGHTLY=1`) |

marker 的第一目标是**行为选择和依赖声明**,与目录分层正交。目录已按分层组织
(`unit/` `contract/` `integration/` `tools/` `e2e/` `helpers/`,见 `test_plan.md`
§3),仅作导航用;`pytest` 的选跑仍以 marker 为准,不靠目录路径。

## 选跑子集

```bash
RUN_REALDATA=1 pytest tests/tools -m "not slow" -q     # 工具回归(自造 fixture)
RUN_E2E=1 pytest tests/e2e -m "e2e" -q                  # e2e wrapper(需对应环境)
pytest tests collector/tests -q                         # 全量本地回归
```

## 新测试放哪

见 `test_plan.md` §11 放置决策表。简版:纯函数→`unit`,schema/signature→
`contract`,多模块 mock 组合→`integration`,脚本 CLI→`tools`,起真实 vLLM→
`e2e`+`gpu`。每个新文件至少属于一种分层 marker;`gpu`/`realdata`/`e2e` 等用于
补充声明依赖。
