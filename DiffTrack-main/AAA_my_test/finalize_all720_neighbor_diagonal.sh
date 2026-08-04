#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
ROOT="/data/gaoya/agent-data/outputs/three_model_all720_neighbor_diagonal_5case"
STATUS_ROOT="${ROOT}/status"
mkdir -p "${STATUS_ROOT}" "${ROOT}/logs"

while true; do
  for model in gt lora baseline; do
    if [[ -f "${STATUS_ROOT}/${model}.failed" ]]; then
      echo "${model} pipeline failed" >&2
      exit 1
    fi
  done
  if [[ -f "${STATUS_ROOT}/gt.complete" && -f "${STATUS_ROOT}/lora.complete" && -f "${STATUS_ROOT}/baseline.complete" ]]; then
    break
  fi
  sleep 30
done

"${PYTHON}" "${PROJECT}/AAA_my_test/aggregate_all_heads_neighbor_diagonal.py" \
  2>&1 | tee "${ROOT}/logs/aggregate.log"
"${PYTHON}" "${PROJECT}/AAA_my_test/render_neighbor_diagonal_ranked_heatmaps.py" \
  2>&1 | tee "${ROOT}/logs/render.log"
printf 'completed=%s\n' "$(date -u +%FT%TZ)" > "${STATUS_ROOT}/pipeline.complete"
echo "ALL720_NEIGHBOR_PIPELINE_COMPLETE"
