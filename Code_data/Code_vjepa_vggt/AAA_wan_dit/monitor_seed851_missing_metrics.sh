#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 RUN_ROOT BATCH_ROOT CPU_WORKERS COMMON_WORKERS VIDEOPHY_WORKERS COSMOS_WORKERS SESSION" >&2
  exit 2
fi

RUN_ROOT="$1"
BATCH_ROOT="$2"
CPU_WORKERS="$3"
COMMON_WORKERS="$4"
VIDEOPHY_WORKERS="$5"
COSMOS_WORKERS="$6"
SESSION="$7"
ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

count_workers() {
  find "${RUN_ROOT}/state" -maxdepth 1 -type f -name "$1" | wc -l
}

count_tasks() {
  grep -c "^$1-" "${RUN_ROOT}/$2" 2>/dev/null || true
}

exec > >(tee -a "${RUN_ROOT}/logs/monitor.log") 2>&1

while true; do
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if [[ "${available_kib}" -lt $((64 * 1024 * 1024)) ]]; then
    echo "[missing-metrics] host memory guard triggered: MemAvailable=${available_kib} KiB; stopping ${SESSION}"
    touch "${RUN_ROOT}/state/memory_guard_triggered"
    tmux kill-session -t "${SESSION}"
    exit 1
  fi
  cpu_done="$(count_workers 'cpu_*.worker_complete')"
  common_done="$(count_workers 'common_*.worker_complete')"
  videophy_done="$(count_workers 'vp_*.worker_complete')"
  cosmos_done="$(count_workers 'cosmos_*.worker_complete')"
  echo "[missing-metrics] mem_available_gib=$((available_kib / 1024 / 1024)) cpu=${cpu_done}/${CPU_WORKERS} common=${common_done}/${COMMON_WORKERS} videophy=${videophy_done}/${VIDEOPHY_WORKERS} cosmos=${cosmos_done}/${COSMOS_WORKERS} completed=$(wc -l < "${RUN_ROOT}/completed_tasks.tsv") failed=$(wc -l < "${RUN_ROOT}/failed_tasks.tsv")"
  if [[ "${cpu_done}" -ge "${CPU_WORKERS}" ]] \
    && [[ "${common_done}" -ge "${COMMON_WORKERS}" ]] \
    && [[ "${videophy_done}" -ge "${VIDEOPHY_WORKERS}" ]] \
    && [[ "${cosmos_done}" -ge "${COSMOS_WORKERS}" ]]; then
    break
  fi
  sleep 30
done

"${PYTHON}" "${ROOT}/summarize_stc_bench_metrics.py" \
  --batch-root "${BATCH_ROOT}"
touch "${RUN_ROOT}/state/all_complete"
echo "[missing-metrics] final summary complete"
exec bash
