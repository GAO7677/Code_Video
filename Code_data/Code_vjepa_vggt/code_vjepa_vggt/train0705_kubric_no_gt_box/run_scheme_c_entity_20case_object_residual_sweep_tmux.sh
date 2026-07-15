#!/usr/bin/env bash
set -euo pipefail

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
WORKER="${BASE}/run_scheme_c_entity_20case_object_residual_worker.sh"
SESSION="${SESSION:-stage1b_scheme_c_entity_20case_scale_sweep_20260715}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

launch_window() {
  local step="$1"
  local scale="$2"
  local gpu="$3"
  local tag="${scale/./p}"
  local name="s${step#step-}_x${tag}_g${gpu}"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${WORKER}' '${step}' '${scale}' '${gpu}'"
}

tmux new-session -d -s "${SESSION}" -n bootstrap \
  "printf 'Launching Scheme-C 20-case residual sweep...\n'; sleep 2"
launch_window step-002500 1.0 0
launch_window step-002500 1.5 1
launch_window step-002500 2.0 2
launch_window step-003500 1.5 3
launch_window step-003500 1.0 4
launch_window step-003500 2.0 5
tmux kill-window -t "${SESSION}:bootstrap" 2>/dev/null || true
tmux select-window -t "${SESSION}:s003500_x1p0_g4"

echo "tmux session: ${SESSION}"
tmux list-windows -t "${SESSION}" -F '#I:#W pane_pid=#{pane_pid} active=#{window_active}'
