#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-wan_s_motion_analysis}"
ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n regions \
  "bash '${ROOT}/run_s_motion_region_worker.sh' 2; exec bash"
tmux new-window -t "${SESSION}" -n coordinator \
  "bash '${ROOT}/run_s_motion_pipeline_coordinator.sh' '${SESSION}'; exec bash"

echo "session=${SESSION}"
echo "regions=GPU2"
echo "feature_workers=GPU0,1,2,3,5,6,7 after strict inventory is complete"
echo "attach: tmux attach -t ${SESSION}"
