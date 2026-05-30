#!/usr/bin/env python3
"""Execute case-driven real-vs-sim vLLM benchmark cases.

The case JSONL is produced by scripts/bench/bench_cases.py. This executor does
not define the benchmark matrix / batch settings — it only runs cases.

Output layout (unchanged):
    <out_root>/<suite>/<case_id>/real_case.txt
    <out_root>/<suite>/<case_id>/sim_case.txt
    <out_root>/<suite>/<case_id>/metrics.json
    <out_root>/<suite>/<case_id>/block_metadata.json
    <out_root>/<suite>/__server_logs/<run_mode>_<model>_tp<n>_<hint>_<mode>.log
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from bench_cases import MODEL_ALIASES  # noqa: E402

REAL_PORT = 8810
SIM_PORT = 8811


@dataclass
class CaseRow:
    server_key: str
    case_id: str
    suite: str
    model_path: str
    tp: int
    ep: int
    hint: str
    mode: str
    concurrency: int
    num_prompts: int
    num_warmups: int
    request_rate: str
    input_len: int
    output_len: int
    prefix: bool
    chunked: bool
    ep_on: bool
    max_model_len: str
    max_num_seqs: str
    max_btoks: str
    gpu_mem: str
    num_gpu_blocks_override: str


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bench_compare.py",
        description="Execute case-driven real-vs-sim vLLM benchmark cases.",
    )
    p.add_argument("--cases", help="case JSONL produced by bench_cases.py")
    p.add_argument("--case-json", help="single inline case JSON (ad-hoc)")
    p.add_argument(
        "--out",
        default=os.environ.get("RESULTS_DIR", "/tmp/bench_compare_results"),
        help="output root",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not args.cases and not args.case_json:
        p.error("provide --cases or --case-json")
    if args.cases and args.case_json:
        p.error("--cases and --case-json are mutually exclusive")
    return args


def _b(value) -> bool:
    return bool(value)


def _s(value) -> str:
    return "" if value is None else str(value)


def load_cases(cases_path: Path) -> list[CaseRow]:
    rows: list[CaseRow] = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        workload = c.get("workload") or {}
        input_len = c.get("input_len", workload.get("input_len"))
        output_len = c.get("output_len", workload.get("output_len"))
        if input_len is None or output_len is None:
            raise SystemExit(f"case {c.get('case_id')} missing input_len/output_len")
        suite = c.get("suite") or c.get("group") or "default"
        model_key = c.get("model") or c.get("model_alias")
        model_path = MODEL_ALIASES.get(model_key, model_key)
        tp = c.get("tp", 1)
        hint = c.get("topology_hint", "concentrated")
        prefix = _b(c.get("prefix_cache", False))
        chunked = _b(c.get("chunked_prefill", False))
        mode = c.get("execution_mode", "cudagraph")
        ep_on = _b(c.get("enable_expert_parallel", False))
        max_model_len = _s(c.get("max_model_len"))
        max_num_seqs = _s(c.get("max_num_seqs"))
        # Fallbacks keep ad-hoc --case-json usable. Generated suite cases set
        # these explicitly in BenchCase, which remains the source of truth.
        max_btoks = _s(c.get("max_num_batched_tokens", 8192))
        gpu_mem = _s(c.get("gpu_mem_util", 0.5))
        num_blocks = _s(c.get("num_gpu_blocks_override"))
        server_key = "|".join(
            [
                str(model_path),
                str(tp),
                hint,
                mode,
                "1" if prefix else "0",
                "1" if chunked else "0",
                "1" if ep_on else "0",
                max_model_len,
                max_num_seqs,
                max_btoks,
                gpu_mem,
                num_blocks,
            ]
        )
        rows.append(
            CaseRow(
                server_key=server_key,
                case_id=c["case_id"],
                suite=suite,
                model_path=str(model_path),
                tp=int(tp),
                ep=int(c.get("ep", 1)),
                hint=hint,
                mode=mode,
                concurrency=int(c.get("concurrency", 1)),
                num_prompts=int(c.get("num_prompts", c.get("concurrency", 1))),
                num_warmups=int(c.get("num_warmups", 1)),
                request_rate=str(c.get("request_rate", "inf")),
                input_len=int(input_len),
                output_len=int(output_len),
                prefix=prefix,
                chunked=chunked,
                ep_on=ep_on,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                max_btoks=max_btoks,
                gpu_mem=gpu_mem,
                num_gpu_blocks_override=num_blocks,
            )
        )
    return rows


def cuda_visible_devices(tp: int, hint: str) -> str:
    table = {
        (2, "concentrated"): "0,1",
        (2, "balanced"): "0,4",
        (4, "concentrated"): "0,1,2,3",
        (4, "balanced"): "0,1,4,5",
    }
    if tp == 1:
        return "0"
    if tp == 8:
        return "0,1,2,3,4,5,6,7"
    if (tp, hint) in table:
        return table[(tp, hint)]
    return ",".join(str(i) for i in range(tp))


def build_server_cmd(row: CaseRow, num_blocks_override: str, port: int) -> list[str]:
    cmd = [
        "vllm", "serve", row.model_path,
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", str(row.tp),
        "--dtype", "bfloat16",
        "--max-num-batched-tokens", row.max_btoks,
        "--gpu-memory-utilization", row.gpu_mem,
        "--max-logprobs", "0",
        "--disable-log-stats",
    ]
    cmd += ["--enable-prefix-caching"] if row.prefix else ["--no-enable-prefix-caching"]
    cmd += ["--enable-chunked-prefill"] if row.chunked else ["--no-enable-chunked-prefill"]
    if row.mode == "eager":
        cmd += ["--enforce-eager"]
    if row.ep_on:
        cmd += ["--enable-expert-parallel"]
    if row.max_model_len:
        cmd += ["--max-model-len", row.max_model_len]
    if row.max_num_seqs:
        cmd += ["--max-num-seqs", row.max_num_seqs]
    if num_blocks_override:
        cmd += ["--num-gpu-blocks-override", str(num_blocks_override)]
    if os.environ.get("DISABLE_ASYNC_SCHED", "off") == "on":
        cmd += ["--no-async-scheduling"]
    return cmd


def build_bench_cmd(row: CaseRow, port: int) -> list[str]:
    return [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--host", "127.0.0.1", "--port", str(port),
        "--model", row.model_path,
        "--dataset-name", "random",
        "--num-prompts", str(row.num_prompts),
        "--num-warmups", str(row.num_warmups),
        "--random-input-len", str(row.input_len),
        "--random-output-len", str(row.output_len),
        "--request-rate", str(row.request_rate),
        "--ignore-eos",
    ]


def parse_kv_cache_info(logfile: Path, block_size: int = 16) -> dict:
    text = logfile.read_text(errors="ignore") if logfile.exists() else ""

    def clean_int(value: str) -> int:
        return int(value.replace(",", ""))

    blocks = None
    source = None
    override = re.findall(r"num_gpu_blocks_override\s*=\s*(\d+)", text)
    if override:
        blocks = int(override[-1])
        source = "override_log"
    else:
        json_matches = re.findall(r'"num_gpu_blocks"\s*:\s*(\d+)', text)
        nonzero = [int(v) for v in json_matches if int(v) > 0]
        if nonzero:
            blocks = nonzero[-1]
            source = "num_gpu_blocks_log"
        if blocks is None:
            assign = re.findall(r"\bnum_gpu_blocks\s*=\s*(\d+)", text)
            nonzero = [int(v) for v in assign if int(v) > 0]
            if nonzero:
                blocks = nonzero[-1]
                source = "num_gpu_blocks_log"

    token_matches = re.findall(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text)
    tokens = clean_int(token_matches[-1]) if token_matches else None

    effective_blocks = None
    effective_source = None
    if tokens is not None and block_size > 0 and tokens % block_size == 0:
        effective_blocks = tokens // block_size
        effective_source = "tokens_div_block_size"
    elif blocks is not None:
        effective_blocks = blocks
        effective_source = source

    conc = re.findall(
        r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x",
        text,
    )
    max_model_len = clean_int(conc[-1][0]) if conc else None
    max_concurrency = float(conc[-1][1]) if conc else None

    return {
        "num_gpu_blocks": blocks,
        "gpu_kv_cache_tokens": tokens,
        "max_model_len": max_model_len,
        "max_concurrency": max_concurrency,
        "source": source,
        "effective_num_gpu_blocks_for_sim": effective_blocks,
        "effective_source": effective_source,
    }


def update_block_metadata(
    out_root: Path, case_id: str, suite: str, run_mode: str,
    info: dict, sim_override: str, blocks_source: str,
) -> None:
    case_dir = out_root / suite / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "block_metadata.json"
    data = json.loads(path.read_text()) if path.exists() else {"case_id": case_id, "kv_blocks": {}}
    data.setdefault("case_id", case_id)
    kv = data.setdefault("kv_blocks", {})
    kv[run_mode] = info
    if sim_override:
        kv["sim_num_gpu_blocks_override"] = int(sim_override)
    if blocks_source:
        kv["blocks_source"] = blocks_source
    path.write_text(json.dumps(data, indent=2) + "\n")


def server_env(run_mode: str, tp: int, hint: str, hw: str) -> dict:
    env = dict(os.environ)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["VLLM_USE_V1"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices(tp, hint)
    env["LLM_INFER_SIM_NUMA_HINT"] = hint
    if run_mode == "real":
        for k in ("VLLM_VIRTUAL_BACKEND", "LLM_INFER_SIM_HW", "LLM_INFER_SIM_TIME_MODE"):
            env.pop(k, None)
    else:
        env["VLLM_VIRTUAL_BACKEND"] = "1"
        env["LLM_INFER_SIM_HW"] = hw
        env.setdefault("LLM_INFER_SIM_TIME_MODE", "realtime")
    return env


def wait_ready(port: int, proc: subprocess.Popen, timeout: int) -> bool:
    import urllib.request

    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return True
        except Exception:
            pass
        if proc.poll() is not None:
            return False
        time.sleep(1)
    return False


def cleanup_server(proc: subprocess.Popen, port: int) -> None:
    user = os.environ.get("USER", "")
    # vLLM v1 splits APIServer + EngineCore into independent processes renamed
    # via prctl to "VLLM::*", so killing only the launcher pid leaves orphan
    # EngineCore holding the port + GPU memory.
    proc.terminate()
    subprocess.run(["pkill", "-P", str(proc.pid)], check=False)
    _force_kill_port(port, user)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(2)
    engine = _pgrep(["-u", user, "-f", "VLLM::"]) if user else _pgrep(["-f", "VLLM::"])
    if engine:
        subprocess.run(["kill", "-9", *engine], check=False)
        time.sleep(1)
    if _listeners(port, user):
        print(f"WARN: still listening on port {port} after kill, force kill")
        _force_kill_port(port, user)
        time.sleep(1)


def _pgrep(args: list[str]) -> list[str]:
    try:
        out = subprocess.run(["pgrep", *args], capture_output=True, text=True, check=False)
        return [p for p in out.stdout.split() if p]
    except FileNotFoundError:
        return []


def _listeners(port: int, user: str) -> list[str]:
    cmd = ["lsof", "-ti", f"tcp:{port}"]
    if user:
        cmd += ["-a", "-u", user]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return [p for p in out.stdout.split() if p]
    except FileNotFoundError:
        return []


def _force_kill_port(port: int, user: str) -> None:
    pids = _listeners(port, user)
    if pids:
        subprocess.run(["kill", "-9", *pids], check=False)


def run_group_mode(
    run_mode: str, server_key: str, port: int, rows: list[CaseRow],
    out_root: Path, hw: str, dry_run: bool, real_blocks: dict,
) -> None:
    group = [r for r in rows if r.server_key == server_key]
    if not group:
        return
    first = group[0]

    num_blocks_override = first.num_gpu_blocks_override
    blocks_source = ""
    if first.num_gpu_blocks_override:
        blocks_source = "explicit"
    elif run_mode == "sim" and server_key in real_blocks:
        num_blocks_override = real_blocks[server_key][0]
        blocks_source = real_blocks[server_key][1] or "real_log"

    model_short = Path(first.model_path).name
    server_dir = out_root / first.suite / "__server_logs"
    server_dir.mkdir(parents=True, exist_ok=True)
    log = server_dir / f"{run_mode}_{model_short}_tp{first.tp}_{first.hint}_{first.mode}.log"

    print(
        f">>> starting {run_mode} server: suite={first.suite} model={model_short} "
        f"tp={first.tp} hint={first.hint} mode={first.mode} port={port}"
    )
    if run_mode == "sim" and num_blocks_override:
        print(f">>> sim num_gpu_blocks_override={num_blocks_override} source={blocks_source}")

    cmd = build_server_cmd(first, num_blocks_override, port)
    env = server_env(run_mode, first.tp, first.hint, hw)

    if dry_run:
        print(
            f"SERVER[{run_mode}] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
            f"{shlex.join(cmd)}"
        )
        for row in group:
            print(f"BENCH[{run_mode}] case={row.case_id} {shlex.join(build_bench_cmd(row, port))}")
        return

    with open(log, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    print(">>> waiting for {} server ready (max 240s)...".format(run_mode))
    if not wait_ready(port, proc, 240):
        print(f"ERROR: {run_mode} server failed to start")
        try:
            print("\n".join(log.read_text(errors="ignore").splitlines()[-50:]))
        except OSError:
            pass
        proc.kill()
        raise RuntimeError(f"{run_mode} server failed to start")

    kv_info = parse_kv_cache_info(log, 16)
    if run_mode == "real":
        if first.num_gpu_blocks_override:
            real_blocks[server_key] = (first.num_gpu_blocks_override, "explicit")
        elif kv_info["effective_num_gpu_blocks_for_sim"]:
            eff = str(kv_info["effective_num_gpu_blocks_for_sim"])
            real_blocks[server_key] = (eff, kv_info["effective_source"] or "real_log")
            raw = kv_info["num_gpu_blocks"]
            if raw and str(raw) != eff:
                print(
                    f">>> real blocks={raw} but sim effective override={eff} "
                    f"source={kv_info['effective_source'] or 'real_log'}"
                )
        elif kv_info["num_gpu_blocks"]:
            real_blocks[server_key] = (
                str(kv_info["num_gpu_blocks"]), kv_info["source"] or "real_log",
            )
        else:
            print(f"WARN: could not parse real num_gpu_blocks from {log}; sim will use its own profile")

    for row in group:
        update_block_metadata(
            out_root, row.case_id, row.suite, run_mode,
            kv_info, num_blocks_override, blocks_source,
        )

    for row in group:
        run_bench_case(run_mode, port, row, out_root, env)

    cleanup_server(proc, port)


def run_bench_case(
    run_mode: str, port: int, row: CaseRow, out_root: Path, env: dict,
) -> None:
    case_dir = out_root / row.suite / row.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    out_file = case_dir / f"{run_mode}_case.txt"
    cmd = build_bench_cmd(row, port)
    print(
        f">>> [{run_mode}] {row.case_id} input={row.input_len} output={row.output_len} "
        f"prompts={row.num_prompts} rate={row.request_rate}"
    )
    with open(out_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False)
    if result.returncode != 0:
        print(f"WARN: bench failed for {run_mode} {row.case_id}")


def write_metrics(out_root: Path, row: CaseRow) -> None:
    case_dir = out_root / row.suite / row.case_id
    extractor = SCRIPT_DIR.parent / "lib" / "_extract_metrics.py"
    metrics = case_dir / "metrics.json"
    err = case_dir / "metrics.err"
    with open(metrics, "w") as out, open(err, "w") as errf:
        result = subprocess.run(
            [sys.executable, str(extractor),
             "--case-id", row.case_id, "--group", row.suite,
             "--case-dir", str(case_dir)],
            stdout=out, stderr=errf, check=False,
        )
    if result.returncode != 0:
        print(f"WARN: metrics extract failed for {row.case_id} (see {err})")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out_root = Path(args.out)
    hw = os.environ.get("HW", "RTX_4090")

    if args.case_json:
        import tempfile

        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        tmp.write(args.case_json + "\n")
        tmp.close()
        cases_path = Path(tmp.name)
    else:
        cases_path = Path(args.cases)
        tmp = None

    try:
        rows = load_cases(cases_path)
        if not rows:
            print(f"ERROR: no cases in {cases_path}", file=sys.stderr)
            return 1

        out_root.mkdir(parents=True, exist_ok=True)
        server_keys = sorted({r.server_key for r in rows})
        print(
            f">>> cases={len(rows)} server_groups={len(server_keys)} "
            f"out={out_root} dry_run={1 if args.dry_run else 0}"
        )

        real_blocks: dict[str, tuple[str, str]] = {}
        for key in server_keys:
            # A single failing group must not abort the whole bench.
            for run_mode, port in (("real", REAL_PORT), ("sim", SIM_PORT)):
                try:
                    run_group_mode(
                        run_mode, key, port, rows, out_root, hw,
                        args.dry_run, real_blocks,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"WARN: {run_mode} group failed (key={key}), continuing: {exc}")

        if not args.dry_run:
            for row in rows:
                write_metrics(out_root, row)

        print(f">>> done. results: {out_root}")
        return 0
    finally:
        if tmp is not None:
            cases_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
