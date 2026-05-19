"""per-op resume checkpoint — JSON 文件, atomic write.

设计:
  - 每个 (framework, op) 一个 checkpoint 文件: `checkpoints/<op>.json`
  - 记录 `done` (case_id set) + `failed` (case_id set)
  - atomic write: 写 tmp + rename, 避免 partial write 损坏
  - load / mark_done / mark_failed / filter_cases 4 个核心操作
  - 默认行为: resume 跳过 done 和 failed; `retry_failed=True` 才重跑 failed

一致性 caveat:
  - 我们的更新顺序是: append_record → mark_done. 中间 crash 会有 jsonl 有数据但
    checkpoint 没记的可能, 下次会重跑同 case 写第二份 — 由 importer dedup 兜底.
  - 此 caveat 优于反过来 (先 mark_done 后 append): 那样 crash 会丢数据.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from collector.schemas import CheckpointState, Case, Framework, OpKind


def load(path: Path, framework: Framework, op: OpKind) -> CheckpointState:
    """读 checkpoint 文件. 不存在或损坏返空 state (do/failed 空)."""
    if not path.exists():
        return CheckpointState(framework=framework, op_kind=op)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return CheckpointState.from_json_dict(d)
    except (json.JSONDecodeError, KeyError, ValueError):
        # 损坏的 checkpoint 不让 scheduler 挂掉, 当空开始 (会重测但安全)
        return CheckpointState(framework=framework, op_kind=op)


def save(path: Path, state: CheckpointState) -> None:
    """原子写: tmp + rename. fsync 保证可见."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_json_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    # fsync tmp 文件
    with open(tmp, "r+b") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def mark_done(path: Path, state: CheckpointState, case_id: str) -> None:
    """标记 case 跑完(成功). 失败列表里的同 id 也一并清掉 (允许重跑成功后扶正)."""
    state.done.add(case_id)
    state.failed.discard(case_id)
    save(path, state)


def mark_failed(path: Path, state: CheckpointState, case_id: str) -> None:
    """标记 case 失败. 不动 done 集合."""
    state.failed.add(case_id)
    save(path, state)


def filter_cases(
    state: CheckpointState,
    cases: list[Case],
    *,
    retry_failed: bool = False,
) -> list[Case]:
    """根据 checkpoint 过滤待跑 case 列表.

    Args:
        retry_failed: True → 重跑 failed (但 done 还是跳); False → 跳 done 和 failed.

    Returns:
        要跑的 case 子集, 顺序跟入参一致.
    """
    skip = set(state.done)
    if not retry_failed:
        skip |= state.failed
    return [c for c in cases if c.case_id not in skip]
