#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_head_role_dose_control_pilot_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${SCRIPT_DIR}/head_role_dose_control_pilot.json"
SESSION=wan_head_role_dose_control
GPUS=(0 1 2 3 5 6 7)

test -s /data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/head_classification/classification_manifest.json
"${PYTHON}" "${SCRIPT_DIR}/run_head_role_dose_control_pilot_worker.py" \
  --config "${CONFIG}" --preflight

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n shell
fi

for gpu in "${GPUS[@]}"; do
  name="pilot_g${gpu}"
  if tmux list-windows -t "${SESSION}" -F '#W' | rg -Fxq "${name}"; then
    echo "window already exists: ${SESSION}:${name}"
    continue
  fi
  command=(
    "${PYTHON}"
    "${SCRIPT_DIR}/run_head_role_dose_control_pilot_worker.py"
    --config "${CONFIG}"
    --gpu "${gpu}"
    --worker-id "${name}"
  )
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-window -d -t "${SESSION}" -n "${name}" \
    "${shell_command}; exec bash"
done

tmux select-window -t "${SESSION}:pilot_g0"
tmux list-windows -t "${SESSION}"
