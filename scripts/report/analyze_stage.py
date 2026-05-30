#!/usr/bin/env python3
"""Compatibility wrapper. Prefer scripts/analyze_bench.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).resolve().parent.parent / "bench" / "analyze_bench.py"
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
