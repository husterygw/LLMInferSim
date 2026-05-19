"""CSV 读写 — 三类 calibration 输出 (详设 §9.4.2 B.3).

schema 对齐 LLMServingSim profiler/perf/<hw>/<model>/<variant>/tp<N>/:

    dense.csv          : layer,tokens,time_us
    per_sequence.csv   : layer,sequences,time_us
    attention.csv      : prefill_chunk,kv_prefill,n_decode,kv_decode,time_us

时间统一为 **微秒** (time_us). Row append-only (sink pattern), 跑断续点也能 resume:
读已有 CSV → 把 shot key set 装入 visited → fire() 跳过已 visited shot.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from llm_infer_sim.calibration.shots import Shot
from llm_infer_sim.calibration.timings import TimingSample


# ---------------------------------------------------------------------------
# Row 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DenseRow:
    layer: str          # canonical
    tokens: int         # num_new_tokens
    time_us: float


@dataclass(frozen=True)
class PerSeqRow:
    layer: str          # canonical (lm_head / sampler)
    sequences: int      # num_decode_seqs
    time_us: float


@dataclass(frozen=True)
class AttnRow:
    prefill_chunk: int
    kv_prefill: int
    n_decode: int
    kv_decode: int
    time_us: float


# ---------------------------------------------------------------------------
# 列名 (跟 LLMServingSim 1:1)
# ---------------------------------------------------------------------------

DENSE_COLS = ("layer", "tokens", "time_us")
PER_SEQ_COLS = ("layer", "sequences", "time_us")
ATTN_COLS = ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us")


# ---------------------------------------------------------------------------
# 写入 (sink, append-only)
# ---------------------------------------------------------------------------

class CsvSink:
    """append-only CSV writer. 文件不存在时写 header, 否则 append.

    用法:
        sink = CsvSink(path, columns=DENSE_COLS)
        sink.write_rows([{"layer": "qkv_proj", "tokens": 128, "time_us": 42.0}])
        # ...
        sink.close()      # 或用 with 上下文
    """

    def __init__(self, path: str | Path, columns: tuple[str, ...]):
        self.path = Path(path)
        self.columns = columns
        self._fh = None
        self._writer = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(self.columns))
        if is_new:
            self._writer.writeheader()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None

    def write_rows(self, rows: Iterable[dict]) -> int:
        if self._writer is None:
            raise RuntimeError("CsvSink 未打开, 用 `with CsvSink(...)`")
        n = 0
        for row in rows:
            self._writer.writerow(row)
            n += 1
        if self._fh is not None:
            self._fh.flush()
        return n


# ---------------------------------------------------------------------------
# 读取 (用于 resume + fit)
# ---------------------------------------------------------------------------

def read_dense(path: str | Path) -> list[DenseRow]:
    rows: list[DenseRow] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(DenseRow(
                layer=row["layer"],
                tokens=int(row["tokens"]),
                time_us=float(row["time_us"]),
            ))
    return rows


def read_per_sequence(path: str | Path) -> list[PerSeqRow]:
    rows: list[PerSeqRow] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(PerSeqRow(
                layer=row["layer"],
                sequences=int(row["sequences"]),
                time_us=float(row["time_us"]),
            ))
    return rows


def read_attention(path: str | Path) -> list[AttnRow]:
    rows: list[AttnRow] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(AttnRow(
                prefill_chunk=int(row["prefill_chunk"]),
                kv_prefill=int(row["kv_prefill"]),
                n_decode=int(row["n_decode"]),
                kv_decode=int(row["kv_decode"]),
                time_us=float(row["time_us"]),
            ))
    return rows


# ---------------------------------------------------------------------------
# Sample → Row 转换 (在 runner 里调)
# ---------------------------------------------------------------------------

def samples_to_dense_rows(shot: Shot, samples: list[TimingSample]) -> list[dict]:
    """dense shot: 每 sample → 一行 (layer, tokens, time_us)."""
    tokens = shot.num_new_tokens
    return [
        {"layer": s.layer, "tokens": tokens, "time_us": s.microseconds}
        for s in samples
    ]


def samples_to_per_seq_rows(shot: Shot, samples: list[TimingSample]) -> list[dict]:
    """per_sequence shot: 每 sample → 一行 (layer, sequences, time_us)."""
    seqs = shot.num_decode_seqs
    return [
        {"layer": s.layer, "sequences": seqs, "time_us": s.microseconds}
        for s in samples
    ]


def samples_to_attn_rows(shot: Shot, samples: list[TimingSample]) -> list[dict]:
    """attention shot: 4D key + time_us; layer 名固定 'attention' 不入列 (4D 已唯一).

    每 shot 该只对 1 个 canonical (`attention`) 匹中, 但保险起见每 sample 都出一行.
    """
    kv_pref = shot.kv_lens_prefill[0] if shot.kv_lens_prefill else 0
    kv_dec = shot.kv_lens_decode[0] if shot.kv_lens_decode else 0
    return [
        {
            "prefill_chunk": shot.prefill_chunk,
            "kv_prefill": kv_pref,
            "n_decode": shot.num_decode_seqs,
            "kv_decode": kv_dec,
            "time_us": s.microseconds,
        }
        for s in samples
    ]


# ---------------------------------------------------------------------------
# Resume 工具: 读已有 CSV, 提取已 visited 的 shot key
# ---------------------------------------------------------------------------

def visited_keys_dense(path: str | Path) -> set[tuple]:
    """dense shot key = (tokens,)."""
    return {(r.tokens,) for r in read_dense(path)}


def visited_keys_per_sequence(path: str | Path) -> set[tuple]:
    return {(r.sequences,) for r in read_per_sequence(path)}


def visited_keys_attention(path: str | Path) -> set[tuple]:
    return {
        (r.prefill_chunk, r.kv_prefill, r.n_decode, r.kv_decode)
        for r in read_attention(path)
    }
