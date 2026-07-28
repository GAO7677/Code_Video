#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_seed851_baseline_bench
RUN_ROOT="${BATCH_ROOT}/run_20260728_dynamic"

count_records() {
  local kind="$1"
  local file="$2"
  awk -F'\t' -v prefix="${kind}-" \
    'index($1, prefix) == 1 {count++} END {print count + 0}' \
    "${RUN_ROOT}/${file}"
}

while true; do
  all_complete=1
  status=()
  for kind in common videophy cosmos; do
    queued="$(wc -l < "${RUN_ROOT}/queues/${kind}.tsv")"
    claimed="$(( $(<"${RUN_ROOT}/queues/${kind}.cursor") - 1 ))"
    completed="$(count_records "${kind}" completed_tasks.tsv)"
    failed="$(count_records "${kind}" failed_tasks.tsv)"
    status+=(
      "${kind}:claimed=${claimed}/${queued},completed=${completed},failed=${failed}"
    )
    if [[ "$((completed + failed))" -lt "${queued}" ]]; then
      all_complete=0
    fi
  done
  echo "[baseline-monitor] ${status[*]}"
  [[ "${all_complete}" -eq 1 ]] && break
  sleep 30
done

"${PYTHON}" "${ROOT}/summarize_stc_bench_metrics.py" \
  --batch-root "${BATCH_ROOT}"
touch "${RUN_ROOT}/state/all_complete"
echo "[baseline-monitor] all queues terminal; summary complete"
exec bash
