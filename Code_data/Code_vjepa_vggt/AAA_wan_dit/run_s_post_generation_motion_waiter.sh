#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_s_post_generation_motion_waiter.sh

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
ANALYSIS_ROOT="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis"
VBENCH_LATEST="${ANALYSIS_ROOT}/vbench_snapshots/latest"
SNAPSHOT_ROOT="${ANALYSIS_ROOT}/incremental_snapshots"
READY="${ANALYSIS_ROOT}/post_generation_motion.complete"
FAILED="${ANALYSIS_ROOT}/post_generation_motion.failed"
LOG_ROOT="${ANALYSIS_ROOT}/post_generation_motion"

mkdir -p "${LOG_ROOT}" "${SNAPSHOT_ROOT}"
rm -f "${READY}" "${FAILED}"

echo "[post-generation-motion] waiting for final VBench"
while true; do
  latest="$(cat "${VBENCH_LATEST}" 2>/dev/null || true)"
  if [[ "$(basename "${latest}")" == final_* && -f "${latest}/run.complete" ]]; then
    break
  fi
  if [[ "$(basename "${latest}")" == final_* && -f "${latest}/run.failed" ]]; then
    touch "${FAILED}"
    echo "[post-generation-motion] final VBench failed: ${latest}" >&2
    exit 1
  fi
  sleep 60
done

echo "[post-generation-motion] building strict motion inventory"
while ! "${PYTHON}" "${ROOT}/build_s_motion_inventory.py" \
  --output-root "${ANALYSIS_ROOT}" \
  > "${LOG_ROOT}/inventory.log" 2>&1; do
  tail -n 1 "${LOG_ROOT}/inventory.log" || true
  sleep 60
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
snapshot="${SNAPSHOT_ROOT}/final_${stamp}"
session="wan_s_motion_incremental_final_${stamp}"
mkdir -p "${snapshot}"
cp "${ANALYSIS_ROOT}/inventory.json" "${snapshot}/inventory.json"
printf '%s\n' "${snapshot}" > "${SNAPSHOT_ROOT}/latest"

echo "[post-generation-motion] launching ${session}"
SESSION="${session}" GPU_LIST="0 3 5" \
  bash "${ROOT}/run_s_motion_incremental_tmux.sh" "${snapshot}" \
  > "${LOG_ROOT}/launch.log" 2>&1

while [[ ! -f "${snapshot}/analysis.complete" && ! -f "${snapshot}/analysis.failed" ]]; do
  sleep 60
done
if [[ -f "${snapshot}/analysis.failed" ]]; then
  touch "${FAILED}"
  echo "[post-generation-motion] motion analysis failed: ${snapshot}" >&2
  exit 1
fi

touch "${READY}"
echo "[post-generation-motion] complete: ${snapshot}"
