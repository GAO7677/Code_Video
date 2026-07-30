#!/usr/bin/env bash
set -euo pipefail

# Run this script on SSH host 118 after dependencies have been synchronized.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/mnt/data/gaoya/agent-data/envs/wan-cu128/bin/python
CONFIG="${SCRIPT_DIR}/head_role_s_dominant_depth_experiment_ssh118.json"
RUNNER="${SCRIPT_DIR}/run_matched_head_subset_ablation_job_ssh118.sh"
SESSION=wan_s_dominant_depth_ssh118
GPUS=(4 5 6 7)

"${PYTHON}" "${SCRIPT_DIR}/run_head_role_dose_control_pilot_worker.py" \
  --config "${CONFIG}" --runner "${RUNNER}" --preflight

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n shell
fi
for gpu in "${GPUS[@]}"; do
  name="remote_g${gpu}"
  if tmux list-windows -t "${SESSION}" -F '#W' | grep -Fxq "${name}"; then
    continue
  fi
  command=(
    "${PYTHON}"
    "${SCRIPT_DIR}/run_head_role_dose_control_pilot_worker.py"
    --config "${CONFIG}"
    --runner "${RUNNER}"
    --gpu "${gpu}"
    --worker-id "${name}"
  )
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-window -d -t "${SESSION}" -n "${name}" \
    "${shell_command}; exec bash"
done
tmux list-windows -t "${SESSION}"
