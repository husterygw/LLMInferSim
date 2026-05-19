"""JSONL append writer — 多进程安全 + fsync.

设计:
  - 每写一条 record 加 fcntl.flock, 多 worker process 并发安全
  - fsync 强制落盘 (NFS / 进程崩溃保护)
  - 错误 record 写到独立 errors/<op>.jsonl, 不阻塞主流程
  - progress.jsonl 是 append-only 审计日志 (一行一次 update)

不在 writer 做的事:
  - 不去重 (writer 是 dumb append, 同 case_id 跑两次会写两行, importer 负责 dedup)
  - 不做 schema 校验 (假定 record 来自 schemas.py dataclass, to_json_dict 已 self-validated)
"""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from collector.schemas import ErrorRecord, ProgressEntry, RawRecord


@contextmanager
def _locked_append(path: Path) -> Iterator:
    """打开 path 追加模式, fcntl.flock 独占锁, yield file object.

    异常安全: 退出 context 时一定 fsync + 释放锁.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a", encoding="utf-8")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield f
        finally:
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass    # fsync 失败不致命 (例 /tmp tmpfs 不支持)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def append_record(path: Path, record: RawRecord) -> None:
    """主输出: 写一条成功 RawRecord 到 <op>.jsonl."""
    with _locked_append(path) as f:
        f.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")


def append_error(path: Path, error: ErrorRecord) -> None:
    """错误输出: 写一条 ErrorRecord 到 errors/<op>.jsonl."""
    with _locked_append(path) as f:
        f.write(json.dumps(error.to_json_dict(), ensure_ascii=False) + "\n")


def append_progress(path: Path, entry: ProgressEntry) -> None:
    """跨 op 总进度: append 到 progress.jsonl (审计日志, 不覆盖)."""
    with _locked_append(path) as f:
        f.write(json.dumps(entry.to_json_dict(), ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[RawRecord]:
    """读 <op>.jsonl, 返 list[RawRecord]. 用于 importer / 测试."""
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(RawRecord.from_json_dict(json.loads(line)))
    return out


def read_errors(path: Path) -> list[ErrorRecord]:
    """读 errors/<op>.jsonl."""
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(ErrorRecord.from_json_dict(json.loads(line)))
    return out
