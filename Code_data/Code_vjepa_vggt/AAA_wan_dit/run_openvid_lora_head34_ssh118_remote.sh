#!/usr/bin/env bash
set -euo pipefail

# Run on SSH host 118. This experiment is intentionally restricted to GPU6/7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/mnt/data/gaoya/agent-data/envs/wan-cu128/bin/python
CONFIG="${SCRIPT_DIR}/head_role_openvid_lora_head34_experiment_ssh118.json"
GROUP_RUNNER="${SCRIPT_DIR}/run_openvid_lora_matched_head_job_ssh118.sh"
BASELINE_RUNNER="${SCRIPT_DIR}/run_openvid_lora_baseline_job_ssh118.sh"
SESSION=wan_openvid_lora_head34_ssh118

"${PYTHON}" "${SCRIPT_DIR}/run_head_role_dose_control_pilot_worker.py" \
  --config "${CONFIG}" --runner "${GROUP_RUNNER}" --preflight

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "${SESSION} already exists"
  tmux list-windows -t "${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" -n openvid_g6
gpu6_command=(
  bash -lc
  "${PYTHON} '${SCRIPT_DIR}/run_openvid_lora_baseline_worker.py' \
    --config '${CONFIG}' --runner '${BASELINE_RUNNER}' --gpu 6 && \
   ${PYTHON} '${SCRIPT_DIR}/run_head_role_dose_control_pilot_worker.py' \
    --config '${CONFIG}' --runner '${GROUP_RUNNER}' \
    --gpu 6 --worker-id openvid_remote_g6"
)
printf -v gpu6_shell_command '%q ' "${gpu6_command[@]}"
tmux send-keys -t "${SESSION}:openvid_g6" "${gpu6_shell_command}" C-m

gpu7_command=(
  "${PYTHON}"
  "${SCRIPT_DIR}/run_head_role_dose_control_pilot_worker.py"
  --config "${CONFIG}"
  --runner "${GROUP_RUNNER}"
  --gpu 7
  --worker-id openvid_remote_g7
)
printf -v gpu7_shell_command '%q ' "${gpu7_command[@]}"
tmux new-window -d -t "${SESSION}" -n openvid_g7 \
  "${gpu7_shell_command}; exec bash"
tmux list-windows -t "${SESSION}"
