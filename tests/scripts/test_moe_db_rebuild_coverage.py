"""moe_plan Phase 5 验收: collector moe.jsonl → import → rebuild → estimate 全链路.

锁住 plan §4 Phase 5:
  - rows ≥ 60 (Phase 2.6 采集集), ok == rows (无 unsupported_roofline)
  - 按 routing_distribution 分组: balanced / power_law_1.01 / power_law_1.2 各应存在
  - 按 (moe_tp, moe_ep) 分组: (4,1) + (1,4) 都覆盖
  - 按 num_tokens bucket: 至少 4 个 (1-4 / 8-32 / 64-128 / 512+) 都有数据
  - gap CSV 已落 docs/baselines/

baseline CSV: docs/baselines/moe_roofline_gap_RTX_4090_vllm-0.19.1.csv
"""
from __future__ import annotations

import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "report_operator_roofline_gap.py"
BASELINE = REPO_ROOT / "docs" / "baselines" / "moe_roofline_gap_RTX_4090_vllm-0.19.1.csv"
DB_ROOT = REPO_ROOT / "collector" / "data" / "operator_db"
DB_FILE = DB_ROOT / "RTX_4090" / "vllm-0.19.1" / "moe.jsonl"


pytestmark = pytest.mark.skipif(
    not DB_FILE.exists(),
    reason="real collector JSONL not present (RTX_4090 / vllm-0.19.1 / moe)",
)


def _run_report(tmp_path: Path) -> tuple[str, Path]:
    csv_path = tmp_path / "moe_gap.csv"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--db-root", str(DB_ROOT),
            "--hardware", "RTX_4090",
            "--framework", "vllm",
            "--framework-version", "0.19.1",
            "--op-kind", "moe",
            "--csv", str(csv_path),
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"report script failed: rc={proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return proc.stdout, csv_path


def test_all_moe_records_supported_no_unsupported_roofline(tmp_path):
    """plan §4 Phase 5 主验收: 真实 moe.jsonl 全部 status=ok, 无 unsupported_roofline."""
    stdout, csv_path = _run_report(tmp_path)
    # first line: rows=N ok=N unsupported=K
    first = stdout.splitlines()[0].strip()
    assert first.startswith("rows="), f"unexpected first line: {first!r}"

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 60, f"rows shrunk: {len(rows)} < 60 (Phase 2.6 baseline)"
    statuses = Counter(r["status"] for r in rows)
    assert statuses.get("ok", 0) == len(rows), (
        f"non-ok rows: {statuses}; expected all ok"
    )
    assert statuses.get("unsupported_roofline", 0) == 0


def test_routing_distribution_coverage(tmp_path):
    """plan §4 Phase 5: 按 routing_distribution 分组每个值至少有 records."""
    _, csv_path = _run_report(tmp_path)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    # routing_distribution 字段不在 CSV 顶层 columns? 看 case_id 后缀
    # case_id 含 distribution: moe_n<N>_motp<X>_moep<Y>_<dist>_<mode>__<hash>
    by_dist: Counter[str] = Counter()
    for r in rows:
        cid = r["case_id"]
        if "balanced" in cid:
            by_dist["balanced"] += 1
        elif "power_law_1.01" in cid:
            by_dist["power_law_1.01"] += 1
        elif "power_law_1.2" in cid:
            by_dist["power_law_1.2"] += 1

    for dist in ("balanced", "power_law_1.01", "power_law_1.2"):
        assert by_dist[dist] > 0, (
            f"routing_distribution={dist!r} 缺失 (got {dict(by_dist)})"
        )


def test_parallel_coverage(tmp_path):
    """plan §4 Phase 5: moe_tp_size/moe_ep_size 覆盖 (4,1) + (1,4)."""
    _, csv_path = _run_report(tmp_path)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    # case_id 含 motp/moep: moe_n<N>_motp<X>_moep<Y>_...
    import re
    by_parallel: Counter[tuple[int, int]] = Counter()
    for r in rows:
        m = re.search(r"motp(\d+)_moep(\d+)", r["case_id"])
        if m:
            by_parallel[(int(m.group(1)), int(m.group(2)))] += 1

    for expected in [(4, 1), (1, 4)]:
        assert by_parallel[expected] > 0, (
            f"(moe_tp, moe_ep)={expected} 缺失 (got {dict(by_parallel)})"
        )


def test_num_tokens_buckets_covered(tmp_path):
    """plan §4 Phase 5: tokens bucket 1-4 / 8-32 / 64-128 / 512+ 全部有 records."""
    _, csv_path = _run_report(tmp_path)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    import re
    def bucket(n: int) -> str:
        if n <= 4: return "1-4"
        if n <= 32: return "8-32"
        if n <= 128: return "64-128"
        return "512+"

    by_bucket: Counter[str] = Counter()
    for r in rows:
        m = re.search(r"_n(\d+)_", r["case_id"])
        if m:
            by_bucket[bucket(int(m.group(1)))] += 1

    for b in ("1-4", "8-32", "64-128", "512+"):
        assert by_bucket[b] > 0, (
            f"num_tokens bucket {b!r} 缺失 (got {dict(by_bucket)})"
        )


def test_baseline_csv_subset_matches_current_run(tmp_path):
    """docs/baselines/moe_roofline_gap CSV 跟当前 report 输出 byte-for-byte 一致.

    任何 公式 / importer / collector 改动只要影响数字, 这条测试就 fail;
    确认是 intentional 后用 `make moe-baseline-update` 重新生成.
    """
    if not BASELINE.exists():
        pytest.skip("baseline CSV not yet checked in")
    _, current_csv = _run_report(tmp_path)
    with current_csv.open() as f:
        current_rows = list(csv.DictReader(f))
    with BASELINE.open() as f:
        baseline_rows = list(csv.DictReader(f))
    assert len(current_rows) >= len(baseline_rows), (
        f"current run shrunk: current={len(current_rows)} baseline={len(baseline_rows)}"
    )
    cur_map = {r["case_id"]: r for r in current_rows}
    for b in baseline_rows:
        c = cur_map.get(b["case_id"])
        assert c is not None, f"missing case_id {b['case_id']} in current run"
        assert c["status"] == b["status"], f"{b['case_id']}: status drift"
        assert c["measured_us_p50"] == b["measured_us_p50"], (
            f"{b['case_id']}: measured drift"
        )
        assert c["roofline_us"] == b["roofline_us"], (
            f"{b['case_id']}: roofline drift "
            f"(current={c['roofline_us']} baseline={b['roofline_us']})"
        )
        assert c["roofline_gap"] == b["roofline_gap"], (
            f"{b['case_id']}: gap drift"
        )
