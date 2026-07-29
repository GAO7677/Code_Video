#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:?usage: run_s_motion_pipeline_coordinator.sh TMUX_SESSION}"
ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
OUTPUT="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis"
STATE="${OUTPUT}/state"
LOGS="${OUTPUT}/logs"
GPUS=(0 1 2 3 5 6 7)

mkdir -p "${STATE}" "${LOGS}"
rm -f "${STATE}/pipeline.complete" "${STATE}/pipeline.failed"
echo "[coordinator] waiting for 20 region caches and all generation records"
while true; do
  if [[ -f "${STATE}/regions.failed" ]]; then
    echo "[coordinator] region cache worker failed" >&2
    touch "${STATE}/pipeline.failed"
    exit 1
  fi
  if "${PYTHON}" "${ROOT}/build_s_motion_inventory.py" \
    > "${LOGS}/inventory.latest.log" 2>&1; then
    break
  fi
  "${PYTHON}" "${ROOT}/build_motion_n_analysis_status.py" \
    >> "${LOGS}/coordinator_status.log" 2>&1 || true
  tail -n 1 "${LOGS}/coordinator_status.log" || true
  sleep 60
done

echo "[coordinator] strict inventory complete; starting ${#GPUS[@]} feature shards"
for shard in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard]}"
  tmux new-window -t "${SESSION}" -n "feat-g${gpu}" \
    "bash '${ROOT}/run_s_motion_feature_worker.sh' '${gpu}' '${shard}' '${#GPUS[@]}'; exec bash"
done

while true; do
  complete="$(find "${STATE}" -maxdepth 1 -name 'features_shard_*.complete' -type f | wc -l)"
  failed="$(find "${STATE}" -maxdepth 1 -name 'features_shard_*.failed' -type f | wc -l)"
  features="$(find "${OUTPUT}/features" -mindepth 2 -maxdepth 2 -name metadata.json -type f 2>/dev/null | wc -l)"
  echo "[coordinator] feature_shards=${complete}/${#GPUS[@]} failed=${failed} features=${features}/1760"
  if [[ "${failed}" -gt 0 ]]; then
    touch "${STATE}/pipeline.failed"
    exit 1
  fi
  [[ "${complete}" -eq "${#GPUS[@]}" ]] && break
  sleep 30
done

"${PYTHON}" "${ROOT}/analyze_s_motion.py" 2>&1 | tee "${LOGS}/analysis.log"
touch "${STATE}/pipeline.complete"
echo "[coordinator] pipeline complete"
