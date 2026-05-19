#!/bin/bash
# Driver: 跑 measure_collectives.py 在所有 (n, NUMA) 配置上.
#
# 当前测试机:8× RTX 4090, GPU 0-3 = NUMA 0, GPU 4-7 = NUMA 1
# (用 `nvidia-smi topo -m` 确认; 0-3 互相 PXB, 0-3 vs 4-7 跨 SYS)
#
# 配置覆盖 5 组:
#   same_numa_n2  : GPU 0,1
#   cross_numa_n2 : GPU 0,4
#   same_numa_n4  : GPU 0,1,2,3
#   cross_numa_n4 : GPU 0,1,4,5 (2 from each NUMA)
#   full_n8       : GPU 0-7 (always cross-NUMA)

set -e

source /data/ygw/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate llm_sim 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYSCRIPT="$SCRIPT_DIR/measure_collectives.py"
OUT="${OUT:-/tmp/collective_bench.jsonl}"

# 清空 output, 重新累计
: > "$OUT"
echo "OUTPUT: $OUT"

# config: gpus label
CONFIGS=(
  "0,1            same_numa_n2"
  "0,4            cross_numa_n2"
  "0,1,2,3        same_numa_n4"
  "0,1,4,5        cross_numa_n4"
  "0,1,2,3,4,5,6,7  full_n8"
)

PORT=29504
for cfg in "${CONFIGS[@]}"; do
  read -r gpus label <<< "$cfg"
  n=$(echo "$gpus" | tr ',' '\n' | wc -l)
  echo
  echo ">>>>>>>>>> running label=$label  GPUs=$gpus  n=$n <<<<<<<<<<"
  CUDA_VISIBLE_DEVICES="$gpus" \
    torchrun --nproc_per_node="$n" --master_port="$PORT" \
    "$PYSCRIPT" \
    --out "$OUT" \
    --label "$label" 2>&1 | tail -180
  PORT=$((PORT + 1))
  echo "    (PORT was $((PORT - 1)))"
done

echo
echo "================================================"
echo "DONE. raw measurements: $OUT"
echo "lines: $(wc -l < "$OUT")"
echo
echo "Run analyze_collectives.py next to compare with cost model predictions."
