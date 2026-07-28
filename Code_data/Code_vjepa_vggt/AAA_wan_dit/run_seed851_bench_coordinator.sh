#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 RUN_ROOT BATCH_ROOT COMMON_WORKERS VIDEOPHY_WORKERS COSMOS_WORKERS" >&2
  exit 2
fi

RUN_ROOT="$1"
BATCH_ROOT="$2"
COMMON_WORKERS="$3"
VIDEOPHY_WORKERS="$4"
COSMOS_WORKERS="$5"
ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

count_state() {
  find "${RUN_ROOT}/state" -maxdepth 1 -name "$1" -type f | wc -l
}

count_tasks() {
  grep -c "^$1-" "${RUN_ROOT}/$2" 2>/dev/null || true
}

while true; do
  common_done="$(count_state 'common_*.worker_complete')"
  videophy_done="$(count_state 'videophy_*.worker_complete')"
  cosmos_done="$(count_state 'cosmos_*.worker_complete')"
  echo "[coordinator] common=${common_done}/${COMMON_WORKERS} done=$(count_tasks common completed_tasks.tsv) failed=$(count_tasks common failed_tasks.tsv) | videophy=${videophy_done}/${VIDEOPHY_WORKERS} done=$(count_tasks videophy completed_tasks.tsv) failed=$(count_tasks videophy failed_tasks.tsv) | cosmos=${cosmos_done}/${COSMOS_WORKERS} done=$(count_tasks cosmos completed_tasks.tsv) failed=$(count_tasks cosmos failed_tasks.tsv)"
  if [[ "${common_done}" -ge "${COMMON_WORKERS}" ]] \
    && [[ "${videophy_done}" -ge "${VIDEOPHY_WORKERS}" ]] \
    && [[ "${cosmos_done}" -ge "${COSMOS_WORKERS}" ]]; then
    break
  fi
  sleep 30
done

"${PYTHON}" "${ROOT}/summarize_stc_bench_metrics.py" \
  --batch-root "${BATCH_ROOT}"
touch "${RUN_ROOT}/state/all_complete"
echo "[coordinator] final summary complete"
exec bash
