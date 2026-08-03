#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${ROOT}/AAA_my_test/run_pck_extreme_head_zero_ablation_worker.py"
INPUT="/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
OUTPUT="/data/gaoya/agent-data/outputs/pck_extreme30_all720_head_zero_ablation_test5"
mkdir -p "${OUTPUT}/logs"

mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
  awk -F, '{gsub(/ /, "", $0); if (($2 + 0) < 2000 && ($3 + 0) < 20) print $1}')
if ((${#GPUS[@]} == 0)); then
  echo "No GPU satisfies memory<2GB and utilization<20%." >&2
  exit 1
fi
if ((${#GPUS[@]} > 20)); then
  GPUS=("${GPUS[@]:0:20}")
fi
NUM_SHARDS=${#GPUS[@]}

run_model() {
  local gpu="$1" model="$2" shard="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${WORKER}" \
    --model "${model}" --input-json-list "${INPUT}" --output-root "${OUTPUT}" \
    --shard-index "${shard}" --num-shards "${NUM_SHARDS}" --device cuda:0 \
    --ranking-pool all720
}

PIDS=()
for lane in "${!GPUS[@]}"; do
  gpu="${GPUS[$lane]}"
  if ((lane % 2 == 0)); then
    (run_model "${gpu}" baseline "${lane}"; run_model "${gpu}" lora "${lane}") \
      >"${OUTPUT}/logs/lane${lane}_gpu${gpu}.log" 2>&1 &
  else
    (run_model "${gpu}" lora "${lane}"; run_model "${gpu}" baseline "${lane}") \
      >"${OUTPUT}/logs/lane${lane}_gpu${gpu}.log" 2>&1 &
  fi
  PIDS+=("$!")
  echo "lane ${lane}: GPU ${gpu}, test_5 shard ${lane}/${NUM_SHARDS}"
done

trap 'kill "${PIDS[@]}" 2>/dev/null || true' INT TERM EXIT
STATUS=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || STATUS=$?
done
trap - INT TERM EXIT
if ((STATUS == 0)); then
  date -u +'{"status":"complete","finished_utc":"%Y-%m-%dT%H:%M:%SZ"}' >"${OUTPUT}/RUN_COMPLETE.json"
fi
exit "${STATUS}"
