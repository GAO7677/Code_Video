#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-wan_s_motion_incremental}"
ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
OUTPUT="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis"
LATEST="${OUTPUT}/incremental_snapshots/latest"
REPORT="/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/multiseed/motion-n-analysis/partial"
GPUS=(0 1 2 3 5 6 7)

SNAPSHOT_DIR="${1:-$(cat "${LATEST}")}"
if [[ ! -f "${SNAPSHOT_DIR}/inventory.json" ]]; then
  echo "missing snapshot inventory: ${SNAPSHOT_DIR}/inventory.json" >&2
  exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "${SNAPSHOT_DIR}/state" "${SNAPSHOT_DIR}/logs"
rm -f "${SNAPSHOT_DIR}/analysis.complete" "${SNAPSHOT_DIR}/analysis.failed"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do complete=\$(find '${SNAPSHOT_DIR}/state' -maxdepth 1 -name 'shard_*.complete' -type f | wc -l); failed=\$(find '${SNAPSHOT_DIR}/state' -maxdepth 1 -name 'shard_*.failed' -type f | wc -l); printf '[incremental-coordinator] shards=%s/${#GPUS[@]} failed=%s\\n' \"\${complete}\" \"\${failed}\"; if [ \"\${failed}\" -gt 0 ]; then touch '${SNAPSHOT_DIR}/analysis.failed'; break; fi; [ \"\${complete}\" -eq '${#GPUS[@]}' ] && break; sleep 30; done; if [ ! -f '${SNAPSHOT_DIR}/analysis.failed' ]; then '${PYTHON}' '${ROOT}/analyze_s_motion.py' --inventory '${SNAPSHOT_DIR}/inventory.json' --output-root '${OUTPUT}' --report-dir '${REPORT}' --bootstrap-samples 1000 2>&1 | tee '${SNAPSHOT_DIR}/logs/analysis.log' && touch '${SNAPSHOT_DIR}/analysis.complete' || touch '${SNAPSHOT_DIR}/analysis.failed'; fi; '${PYTHON}' '${ROOT}/build_motion_n_analysis_status.py'; exec bash"

for shard in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard]}"
  tmux new-window -t "${SESSION}" -n "g${gpu}" \
    "bash '${ROOT}/run_s_motion_incremental_worker.sh' '${gpu}' '${shard}' '${#GPUS[@]}' '${SNAPSHOT_DIR}'; exec bash"
done

tmux select-window -t "${SESSION}:coordinator"
echo "session=${SESSION}"
echo "snapshot=${SNAPSHOT_DIR}"
echo "gpus=${GPUS[*]}"
echo "report=${REPORT}"
