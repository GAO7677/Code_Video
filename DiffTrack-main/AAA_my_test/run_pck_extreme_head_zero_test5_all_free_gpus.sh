#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${ROOT}/AAA_my_test/run_pck_extreme_head_zero_ablation_worker.py"
INPUT="/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
OUTPUT="/data/gaoya/agent-data/outputs/pck_top30_bottom30_head_zero_ablation_test5"
mkdir -p "${OUTPUT}/logs"

run_model() {
  local gpu="$1" model="$2" shard="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${WORKER}" \
    --model "${model}" --input-json-list "${INPUT}" --output-root "${OUTPUT}" \
    --shard-index "${shard}" --num-shards 5 --device cuda:0 \
    >"${OUTPUT}/logs/${model}_shard${shard}_gpu${gpu}.log" 2>&1
}

pids=()
for gpu in 0 1 2 3 4; do
  if (( gpu % 2 == 0 )); then
    (run_model "${gpu}" baseline "${gpu}"; run_model "${gpu}" lora "${gpu}") &
  else
    (run_model "${gpu}" lora "${gpu}"; run_model "${gpu}" baseline "${gpu}") &
  fi
  pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM EXIT

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
trap - INT TERM EXIT
exit "${status}"
