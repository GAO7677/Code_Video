#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
ANALYSIS_ROOT="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis"
CURRENT="$(cat "${ANALYSIS_ROOT}/vbench_snapshots/latest")"
FINAL_INVENTORY_ROOT="${ANALYSIS_ROOT}/vbench_final_inventory"
LATEST="${ANALYSIS_ROOT}/vbench_snapshots/latest"
LOGS="${ANALYSIS_ROOT}/vbench_final_waiter"

mkdir -p "${FINAL_INVENTORY_ROOT}" "${LOGS}"
echo "[vbench-final-waiter] waiting for current snapshot: ${CURRENT}"
while [[ ! -f "${CURRENT}/run.complete" && ! -f "${CURRENT}/run.failed" ]]; do
  completed="$(wc -l < "${CURRENT}/completed_tasks.tsv")"
  failed="$(wc -l < "${CURRENT}/failed_tasks.tsv")"
  echo "[vbench-final-waiter] current completed=${completed} failed=${failed}"
  sleep 60
done

echo "[vbench-final-waiter] waiting for strict full inventory"
while ! "${PYTHON}" "${ROOT}/build_s_motion_inventory.py" \
  --output-root "${FINAL_INVENTORY_ROOT}" \
  > "${LOGS}/inventory.log" 2>&1; do
  tail -n 1 "${LOGS}/inventory.log" || true
  sleep 60
done

SNAPSHOT="${ANALYSIS_ROOT}/vbench_snapshots/final_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${SNAPSHOT}"
"${PYTHON}" "${ROOT}/build_s_vbench_snapshot.py" \
  --inventory "${FINAL_INVENTORY_ROOT}/inventory.json" \
  --output-root "${SNAPSHOT}" \
  2>&1 | tee "${LOGS}/final_snapshot.log"
printf '%s\n' "${SNAPSHOT}" > "${LATEST}"

tasks="$(wc -l < "${SNAPSHOT}/queues/gpu_common.tsv")"
if [[ "${tasks}" -eq 0 ]]; then
  touch "${SNAPSHOT}/run.complete"
  echo "[vbench-final-waiter] no missing VBench tasks"
  exit 0
fi

echo "[vbench-final-waiter] launching final tasks=${tasks}"
SESSION=wan_s_vbench_final \
  bash "${ROOT}/run_s_vbench_snapshot_tmux.sh" "${SNAPSHOT}"
