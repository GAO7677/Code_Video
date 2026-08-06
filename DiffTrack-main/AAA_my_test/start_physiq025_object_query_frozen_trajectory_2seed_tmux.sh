#!/usr/bin/env bash
set -euo pipefail

SESSION="physiq025_frozen_trajectory_2seed"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"

tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}"
  exit 0
}

tmux new-session -d -s "${SESSION}" -n seed13161 \
  "cd '${DIFFTRACK}' && ./AAA_my_test/run_physiq025_object_query_frozen_trajectory_wait_gpu.sh 13161"
tmux new-window -t "${SESSION}" -n seed16342 \
  "cd '${DIFFTRACK}' && ./AAA_my_test/run_physiq025_object_query_frozen_trajectory_wait_gpu.sh 16342"
echo "started tmux session: ${SESSION}"
echo "attach: tmux attach -t ${SESSION}"
