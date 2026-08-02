#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_auto_stop_no_object_step3000_tmux.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
SESSION="${SESSION:-wan_train_full_sa_no_object_gpu27}"
WINDOW="${WINDOW:-noobj_step3000_auto}"
LOG_ROOT="${LOG_ROOT:-/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/automation/no_object_step3000}"
LOG_PATH="${LOG_ROOT}/automation.log"

mkdir -p "${LOG_ROOT}"
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n shell
fi
if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${WINDOW}"; then
  echo "Automation window already exists: ${SESSION}:${WINDOW}" >&2
  exit 1
fi

command=(
  env PYTHONNOUSERSITE=1
  "${PYTHON}"
  "${SCRIPT_DIR}/auto_stop_no_object_step3000_and_drain_evals.py"
)
printf -v command_shell '%q ' "${command[@]}"
tmux new-window -t "${SESSION}" -n "${WINDOW}" \
  "${command_shell}2>&1 | tee -a '${LOG_PATH}'; status=\${PIPESTATUS[0]}; echo AUTOMATION_EXIT=\${status}; exec bash"

echo "tmux=${SESSION}:${WINDOW}"
echo "log=${LOG_PATH}"
echo "status=${LOG_ROOT}/status.json"
