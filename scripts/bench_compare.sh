#!/bin/bash
# Execute case-driven real-vs-sim vLLM benchmark cases.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASES_JSONL=""
CASE_JSON=""
OUT_ROOT="${RESULTS_DIR:-/tmp/bench_compare_results}"
DRY_RUN=0
HW="${HW:-RTX_4090}"
CONDA_ENV="${CONDA_ENV-llm_sim}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bench_compare.sh --cases /tmp/cases.jsonl --out /tmp/bench_out
  bash scripts/bench_compare.sh --case-json '{"case_id":"debug",...}' --out /tmp/bench_out
  bash scripts/bench_compare.sh --cases /tmp/cases.jsonl --out /tmp/bench_out --dry-run

The case JSONL is produced by scripts/bench_cases.py. bench_compare.sh is an
executor only: it does not define benchmark matrix / batch settings.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cases) CASES_JSONL="$2"; shift 2 ;;
    --case-json) CASE_JSON="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$CASES_JSONL" ] && [ -z "$CASE_JSON" ]; then
  echo "ERROR: provide --cases or --case-json" >&2
  usage
  exit 1
fi

if [ -n "$CASES_JSONL" ] && [ -n "$CASE_JSON" ]; then
  echo "ERROR: --cases and --case-json are mutually exclusive" >&2
  exit 1
fi

if [ -n "$CONDA_ENV" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -z "$CONDA_BASE" ] && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
  fi
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
fi

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TMP_CASES=""
if [ -n "$CASE_JSON" ]; then
  TMP_CASES="$(mktemp)"
  printf '%s\n' "$CASE_JSON" > "$TMP_CASES"
  CASES_JSONL="$TMP_CASES"
fi
trap 'if [ -n "$TMP_CASES" ]; then rm -f "$TMP_CASES"; fi' EXIT

mkdir -p "$OUT_ROOT"

mapfile -t CASE_LINES < <(python3 - "$CASES_JSONL" "$SCRIPT_DIR" <<'PY'
import json
import sys
from pathlib import Path

cases_path = Path(sys.argv[1])
script_dir = sys.argv[2]
sys.path.insert(0, script_dir)
from bench_cases import MODEL_ALIASES  # noqa: E402

def b(value):
    return "1" if bool(value) else "0"

def s(value):
    return "" if value is None else str(value)

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
    prefix = b(c.get("prefix_cache", False))
    chunked = b(c.get("chunked_prefill", False))
    mode = c.get("execution_mode", "cudagraph")
    ep_on = b(c.get("enable_expert_parallel", False))
    max_model_len = s(c.get("max_model_len"))
    max_num_seqs = s(c.get("max_num_seqs"))
    # Fallbacks keep ad-hoc --case-json usable. Generated suite cases set these
    # fields explicitly in BenchCase, which remains the benchmark source of truth.
    max_btoks = s(c.get("max_num_batched_tokens", 8192))
    gpu_mem = s(c.get("gpu_mem_util", 0.5))
    num_gpu_blocks_override = s(c.get("num_gpu_blocks_override"))
    server_key = "|".join([
        str(model_path), str(tp), hint, mode, prefix, chunked, ep_on,
        max_model_len, max_num_seqs, max_btoks, gpu_mem, num_gpu_blocks_override,
    ])
    fields = [
        server_key,
        c["case_id"],
        suite,
        str(model_path),
        str(tp),
        str(c.get("ep", 1)),
        hint,
        mode,
        str(c.get("concurrency", 1)),
        str(c.get("num_prompts", c.get("concurrency", 1))),
        str(c.get("num_warmups", 1)),
        str(c.get("request_rate", "inf")),
        str(input_len),
        str(output_len),
        prefix,
        chunked,
        ep_on,
        max_model_len,
        max_num_seqs,
        max_btoks,
        gpu_mem,
        num_gpu_blocks_override,
    ]
    print("\x1f".join(fields))
PY
)

if [ "${#CASE_LINES[@]}" -eq 0 ]; then
  echo "ERROR: no cases in $CASES_JSONL" >&2
  exit 1
fi

declare -A REAL_NUM_GPU_BLOCKS_BY_KEY
declare -A REAL_BLOCKS_SOURCE_BY_KEY

set_cuda_visible_devices() {
  local tp=$1
  local hint=$2
  case "${tp}_${hint}" in
    1_*)              export CUDA_VISIBLE_DEVICES="0" ;;
    2_concentrated)   export CUDA_VISIBLE_DEVICES="0,1" ;;
    2_balanced)       export CUDA_VISIBLE_DEVICES="0,4" ;;
    4_concentrated)   export CUDA_VISIBLE_DEVICES="0,1,2,3" ;;
    4_balanced)       export CUDA_VISIBLE_DEVICES="0,1,4,5" ;;
    8_*)              export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" ;;
    *)                export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((tp - 1)))" ;;
  esac
}

extract_kv_cache_info() {
  local logfile=$1 block_size=${2:-16}
  python3 - "$logfile" "$block_size" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
block_size = int(sys.argv[2])
text = path.read_text(errors="ignore") if path.exists() else ""

def clean_int(value: str) -> int:
    return int(value.replace(",", ""))

blocks = None
source = None

override_matches = re.findall(r"num_gpu_blocks_override\s*=\s*(\d+)", text)
if override_matches:
    blocks = int(override_matches[-1])
    source = "override_log"
else:
    json_matches = re.findall(r'"num_gpu_blocks"\s*:\s*(\d+)', text)
    if json_matches:
        nonzero = [int(v) for v in json_matches if int(v) > 0]
        if nonzero:
            blocks = nonzero[-1]
            source = "num_gpu_blocks_log"
    if blocks is None:
        assignment_matches = re.findall(r"\bnum_gpu_blocks\s*=\s*(\d+)", text)
        nonzero = [int(v) for v in assignment_matches if int(v) > 0]
        if nonzero:
            blocks = nonzero[-1]
            source = "num_gpu_blocks_log"

token_matches = re.findall(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text)
tokens = clean_int(token_matches[-1]) if token_matches else None

effective_blocks_for_sim = None
effective_source = None
if tokens is not None and block_size > 0 and tokens % block_size == 0:
    effective_blocks_for_sim = tokens // block_size
    effective_source = "tokens_div_block_size"
elif blocks is not None:
    effective_blocks_for_sim = blocks
    effective_source = source

conc_matches = re.findall(
    r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x",
    text,
)
max_model_len = clean_int(conc_matches[-1][0]) if conc_matches else None
max_concurrency = float(conc_matches[-1][1]) if conc_matches else None

print(json.dumps({
    "num_gpu_blocks": blocks,
    "gpu_kv_cache_tokens": tokens,
    "max_model_len": max_model_len,
    "max_concurrency": max_concurrency,
    "source": source,
    "effective_num_gpu_blocks_for_sim": effective_blocks_for_sim,
    "effective_source": effective_source,
}, separators=(",", ":")))
PY
}

update_block_metadata() {
  local case_id=$1 suite=$2 run_mode=$3 info_json=$4 sim_override=$5 blocks_source=$6
  local case_dir="$OUT_ROOT/$suite/$case_id"
  mkdir -p "$case_dir"
  python3 - "$case_dir/block_metadata.json" "$case_id" "$run_mode" "$info_json" "$sim_override" "$blocks_source" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
case_id = sys.argv[2]
run_mode = sys.argv[3]
info = json.loads(sys.argv[4]) if sys.argv[4] else {}
sim_override = int(sys.argv[5]) if sys.argv[5] else None
blocks_source = sys.argv[6] or None

data = json.loads(path.read_text()) if path.exists() else {"case_id": case_id, "kv_blocks": {}}
data.setdefault("case_id", case_id)
kv = data.setdefault("kv_blocks", {})
kv[run_mode] = info
if sim_override is not None:
    kv["sim_num_gpu_blocks_override"] = sim_override
if blocks_source is not None:
    kv["blocks_source"] = blocks_source
path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

build_server_cmd() {
  local model_path=$1 tp=$2 mode=$3 prefix=$4 chunked=$5 ep_on=$6 max_len=$7 max_seqs=$8 max_btoks=$9 gpu_mem=${10} num_blocks_override=${11} port=${12}

  local -a cmd=(vllm serve "$model_path"
    --host 127.0.0.1 --port "$port"
    --tensor-parallel-size "$tp"
    --dtype bfloat16
    --max-num-batched-tokens "$max_btoks"
    --gpu-memory-utilization "$gpu_mem"
    --max-logprobs 0
    --disable-log-stats)

  if [ "$prefix" = "1" ]; then cmd+=(--enable-prefix-caching); else cmd+=(--no-enable-prefix-caching); fi
  if [ "$chunked" = "1" ]; then cmd+=(--enable-chunked-prefill); else cmd+=(--no-enable-chunked-prefill); fi
  if [ "$mode" = "eager" ]; then cmd+=(--enforce-eager); fi
  if [ "$ep_on" = "1" ]; then cmd+=(--enable-expert-parallel); fi
  if [ -n "$max_len" ]; then cmd+=(--max-model-len "$max_len"); fi
  if [ -n "$max_seqs" ]; then cmd+=(--max-num-seqs "$max_seqs"); fi
  if [ -n "$num_blocks_override" ]; then cmd+=(--num-gpu-blocks-override "$num_blocks_override"); fi
  if [ "${DISABLE_ASYNC_SCHED:-off}" = "on" ]; then cmd+=(--no-async-scheduling); fi

  printf '%q ' "${cmd[@]}"
}

start_server() {
  local run_mode=$1 model_path=$2 tp=$3 hint=$4 mode=$5 prefix=$6 chunked=$7 ep_on=$8 max_len=$9 max_seqs=${10} max_btoks=${11} gpu_mem=${12} num_blocks_override=${13} port=${14} logfile=${15}

  set_cuda_visible_devices "$tp" "$hint"
  export LLM_INFER_SIM_NUMA_HINT="$hint"
  if [ "$run_mode" = "real" ]; then
    unset VLLM_VIRTUAL_BACKEND
    unset LLM_INFER_SIM_HW
    unset LLM_INFER_SIM_TIME_MODE
  else
    export VLLM_VIRTUAL_BACKEND=1
    export LLM_INFER_SIM_HW="$HW"
    export LLM_INFER_SIM_TIME_MODE="${LLM_INFER_SIM_TIME_MODE:-realtime}"
  fi

  local cmd
  cmd="$(build_server_cmd "$model_path" "$tp" "$mode" "$prefix" "$chunked" "$ep_on" "$max_len" "$max_seqs" "$max_btoks" "$gpu_mem" "$num_blocks_override" "$port")"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "SERVER[$run_mode] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $cmd"
    return 0
  fi

  # shellcheck disable=SC2086
  eval "$cmd" > "$logfile" 2>&1 &
  echo $!
}

wait_ready() {
  local port=$1 pid=$2 timeout=$3
  for _ in $(seq 1 "$timeout"); do
    if curl -fsS "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

run_bench_case() {
  local run_mode=$1 port=$2 case_id=$3 suite=$4 model_path=$5 num_prompts=$6 num_warmups=$7 rate=$8 input_len=$9 output_len=${10}
  local case_dir="$OUT_ROOT/$suite/$case_id"
  mkdir -p "$case_dir"
  local out_file="$case_dir/${run_mode}_case.txt"
  local -a cmd=(vllm bench serve
    --backend vllm
    --host 127.0.0.1 --port "$port"
    --model "$model_path"
    --dataset-name random
    --num-prompts "$num_prompts"
    --num-warmups "$num_warmups"
    --random-input-len "$input_len"
    --random-output-len "$output_len"
    --request-rate "$rate"
    --ignore-eos)

  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'BENCH[%s] case=%s ' "$run_mode" "$case_id"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  echo ">>> [$run_mode] $case_id input=$input_len output=$output_len prompts=$num_prompts rate=$rate"
  "${cmd[@]}" > "$out_file" 2>&1
}

write_metrics() {
  local case_id=$1 suite=$2
  local case_dir="$OUT_ROOT/$suite/$case_id"
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  python3 "$SCRIPT_DIR/_extract_metrics.py" \
    --case-id "$case_id" --group "$suite" --case-dir "$case_dir" \
    > "$case_dir/metrics.json" 2> "$case_dir/metrics.err" || \
    echo "WARN: metrics extract failed for $case_id (see $case_dir/metrics.err)"
}

run_group_mode() {
  local run_mode=$1 server_key=$2 port=$3
  local first_line=""
  local line key
  # 注意: 不加 local key 会污染上层 for-loop 的 $key, 让 sim 永远拿到错的 server_key.
  for line in "${CASE_LINES[@]}"; do
    IFS=$'\037' read -r key _rest <<< "$line"
    if [ "$key" = "$server_key" ]; then
      first_line="$line"
      break
    fi
  done
  if [ -z "$first_line" ]; then
    return 0
  fi

  IFS=$'\037' read -r _key _case_id suite model_path tp _ep hint mode _conc _np _nw _rate _i _o prefix chunked ep_on max_len max_seqs max_btoks gpu_mem explicit_blocks_override <<< "$first_line"
  local num_blocks_override="$explicit_blocks_override"
  local blocks_source=""
  if [ -n "$explicit_blocks_override" ]; then
    blocks_source="explicit"
  elif [ "$run_mode" = "sim" ] && [ -n "${REAL_NUM_GPU_BLOCKS_BY_KEY[$server_key]:-}" ]; then
    num_blocks_override="${REAL_NUM_GPU_BLOCKS_BY_KEY[$server_key]}"
    blocks_source="${REAL_BLOCKS_SOURCE_BY_KEY[$server_key]:-real_log}"
  fi

  local model_short
  model_short="$(basename "$model_path")"
  local server_dir="$OUT_ROOT/$suite/__server_logs"
  mkdir -p "$server_dir"
  local log="$server_dir/${run_mode}_${model_short}_tp${tp}_${hint}_${mode}.log"

  echo ">>> starting $run_mode server: suite=$suite model=$model_short tp=$tp hint=$hint mode=$mode port=$port"
  if [ "$run_mode" = "sim" ] && [ -n "$num_blocks_override" ]; then
    echo ">>> sim num_gpu_blocks_override=$num_blocks_override source=$blocks_source"
  fi
  local pid=""
  if [ "$DRY_RUN" -eq 1 ]; then
    start_server "$run_mode" "$model_path" "$tp" "$hint" "$mode" "$prefix" "$chunked" "$ep_on" "$max_len" "$max_seqs" "$max_btoks" "$gpu_mem" "$num_blocks_override" "$port" "$log"
  else
    pid=$(start_server "$run_mode" "$model_path" "$tp" "$hint" "$mode" "$prefix" "$chunked" "$ep_on" "$max_len" "$max_seqs" "$max_btoks" "$gpu_mem" "$num_blocks_override" "$port" "$log")
    echo ">>> waiting for $run_mode server ready (max 240s)..."
    if ! wait_ready "$port" "$pid" 240; then
      echo "ERROR: $run_mode server failed to start"
      tail -50 "$log" || true
      kill -9 "$pid" 2>/dev/null || true
      return 1
    fi

    local kv_info kv_blocks kv_source kv_sim_blocks kv_sim_source
    kv_info="$(extract_kv_cache_info "$log" 16)"
    kv_blocks="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("num_gpu_blocks") or "")' "$kv_info")"
    kv_source="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("source") or "")' "$kv_info")"
    kv_sim_blocks="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("effective_num_gpu_blocks_for_sim") or "")' "$kv_info")"
    kv_sim_source="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("effective_source") or "")' "$kv_info")"
    if [ "$run_mode" = "real" ]; then
      if [ -n "$explicit_blocks_override" ]; then
        REAL_NUM_GPU_BLOCKS_BY_KEY[$server_key]="$explicit_blocks_override"
        REAL_BLOCKS_SOURCE_BY_KEY[$server_key]="explicit"
      elif [ -n "$kv_sim_blocks" ]; then
        REAL_NUM_GPU_BLOCKS_BY_KEY[$server_key]="$kv_sim_blocks"
        REAL_BLOCKS_SOURCE_BY_KEY[$server_key]="${kv_sim_source:-real_log}"
        if [ -n "$kv_blocks" ] && [ "$kv_blocks" != "$kv_sim_blocks" ]; then
          echo ">>> real blocks=$kv_blocks but sim effective override=$kv_sim_blocks source=${kv_sim_source:-real_log}"
        fi
      elif [ -n "$kv_blocks" ]; then
        REAL_NUM_GPU_BLOCKS_BY_KEY[$server_key]="$kv_blocks"
        REAL_BLOCKS_SOURCE_BY_KEY[$server_key]="${kv_source:-real_log}"
      else
        echo "WARN: could not parse real num_gpu_blocks from $log; sim will use its own profile"
      fi
    fi

    local meta_case_id meta_suite meta_key
    for line in "${CASE_LINES[@]}"; do
      IFS=$'\037' read -r meta_key meta_case_id meta_suite _rest <<< "$line"
      if [ "$meta_key" != "$server_key" ]; then
        continue
      fi
      update_block_metadata "$meta_case_id" "$meta_suite" "$run_mode" "$kv_info" "$num_blocks_override" "$blocks_source"
    done
  fi

  local case_id model_path num_prompts num_warmups rate input_len output_len
  for line in "${CASE_LINES[@]}"; do
    IFS=$'\037' read -r key case_id suite model_path _tp _ep _hint _mode _conc num_prompts num_warmups rate input_len output_len _prefix _chunked _ep_on _max_len _max_seqs _max_btoks _gpu_mem _blocks_override <<< "$line"
    if [ "$key" != "$server_key" ]; then
      continue
    fi
    run_bench_case "$run_mode" "$port" "$case_id" "$suite" "$model_path" "$num_prompts" "$num_warmups" "$rate" "$input_len" "$output_len" || \
      echo "WARN: bench failed for $run_mode $case_id"
  done

  if [ "$DRY_RUN" -eq 0 ]; then
    # vLLM v1 拆 APIServer + EngineCore 多个独立进程, 用 prctl 改名为 "VLLM::*",
    # 因此 `kill $pid` + `pkill -P $pid` 只杀主进程 + 直接子进程, EngineCore 会
    # 变成孤儿继续占 port + GPU 显存. Linux SO_REUSEPORT 让新旧 server 同时绑
    # 同 port, 请求被 OS round-robin, 导致 sim 完全错误.
    kill "$pid" 2>/dev/null || true
    pkill -P "$pid" 2>/dev/null || true
    # 兜底1: 杀监听本 port 的所有自己进程 (APIServer)
    local stragglers
    stragglers="$(lsof -ti tcp:"$port" -a -u "$USER" 2>/dev/null || true)"
    if [ -n "$stragglers" ]; then
      echo "$stragglers" | xargs -r kill -9 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    sleep 2
    # 兜底2: 按进程名清 VLLM::EngineCore / Worker 等 (EngineCore 不绑 port, lsof
    # 抓不到. 用进程名匹配 prctl 改后的名字)
    local engine_pids
    engine_pids="$(pgrep -u "$USER" -f "VLLM::" 2>/dev/null || true)"
    if [ -n "$engine_pids" ]; then
      echo "$engine_pids" | xargs -r kill -9 2>/dev/null || true
      sleep 1
    fi
    # 验证: port 释放 + 无 VLLM:: 残留
    stragglers="$(lsof -ti tcp:"$port" -a -u "$USER" 2>/dev/null || true)"
    if [ -n "$stragglers" ]; then
      echo "WARN: still listening on port $port after kill, force kill"
      echo "$stragglers" | xargs -r kill -9 2>/dev/null || true
      sleep 1
    fi
  fi
}

mapfile -t SERVER_KEYS < <(printf '%s\n' "${CASE_LINES[@]}" | awk -F $'\037' '{print $1}' | sort -u)

echo ">>> cases=${#CASE_LINES[@]} server_groups=${#SERVER_KEYS[@]} out=$OUT_ROOT dry_run=$DRY_RUN"

for key in "${SERVER_KEYS[@]}"; do
  # 单个 server group 启动失败不应该 abort 整个 bench (其它 group 应继续).
  # set -e 在前, 必须用 || true 吞掉失败.
  run_group_mode real "$key" 8810 || echo "WARN: real group failed (key=$key), continuing"
  run_group_mode sim "$key" 8811 || echo "WARN: sim group failed (key=$key), continuing"
done

for line in "${CASE_LINES[@]}"; do
  IFS=$'\037' read -r _key case_id suite _rest <<< "$line"
  write_metrics "$case_id" "$suite"
done

echo ">>> done. results: $OUT_ROOT"
