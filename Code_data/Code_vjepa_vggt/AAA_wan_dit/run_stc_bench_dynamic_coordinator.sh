#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 RUN_ROOT BATCH_ROOT REPORT_ROOT COMMON_WORKERS VIDEOPHY_WORKERS COSMOS_WORKERS" >&2
  exit 2
fi

RUN_ROOT="$1"
BATCH_ROOT="$2"
REPORT_ROOT="$3"
COMMON_WORKERS="$4"
VIDEOPHY_WORKERS="$5"
COSMOS_WORKERS="$6"
ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

count_state() {
  find "${RUN_ROOT}/state" -maxdepth 1 -name "$1" -type f | wc -l
}

count_tasks() {
  grep -c "^$1-" "${RUN_ROOT}/$2" 2>/dev/null || true
}

while true; do
  workers="$(count_state 'common_*.worker_complete')"
  claimed="$(( $(<"${RUN_ROOT}/queues/common.cursor") - 1 ))"
  echo "[coordinator] common workers=${workers}/${COMMON_WORKERS} claimed=${claimed}/147 completed=$(count_tasks common completed_tasks.tsv) failed=$(count_tasks common failed_tasks.tsv)"
  [[ "${workers}" -ge "${COMMON_WORKERS}" ]] && break
  sleep 30
done
touch "${RUN_ROOT}/common.ready"

while true; do
  workers="$(count_state 'videophy_*.worker_complete')"
  echo "[coordinator] videophy workers=${workers}/${VIDEOPHY_WORKERS} completed=$(count_tasks videophy completed_tasks.tsv) failed=$(count_tasks videophy failed_tasks.tsv)"
  [[ "${workers}" -ge "${VIDEOPHY_WORKERS}" ]] && break
  sleep 30
done
touch "${RUN_ROOT}/videophy.ready"

while true; do
  workers="$(count_state 'cosmos_*.worker_complete')"
  echo "[coordinator] cosmos workers=${workers}/${COSMOS_WORKERS} completed=$(count_tasks cosmos completed_tasks.tsv) failed=$(count_tasks cosmos failed_tasks.tsv)"
  [[ "${workers}" -ge "${COSMOS_WORKERS}" ]] && break
  sleep 30
done

"${PYTHON}" "${ROOT}/summarize_stc_bench_metrics.py" \
  --batch-root "${BATCH_ROOT}"
"${PYTHON}" "${ROOT}/render_stc_bench_metric_report.py" \
  --batch-root "${BATCH_ROOT}" \
  --output-dir "${REPORT_ROOT}"
echo "[coordinator] final summary and report complete"
exec bash
