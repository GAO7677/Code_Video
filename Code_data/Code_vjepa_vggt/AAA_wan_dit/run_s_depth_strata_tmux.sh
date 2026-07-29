#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_s_depth_strata_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${SCRIPT_DIR}/s_depth_strata_experiment.json"
SESSION=wan_s_depth_strata
GPUS=(0 1 2 3 5 6 7)

"${PYTHON}" "${SCRIPT_DIR}/build_s_depth_strata_manifest.py"
"${PYTHON}" "${SCRIPT_DIR}/run_head_role_dose_control_pilot_worker.py" \
  --config "${CONFIG}" --preflight

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n shell
fi

if ! tmux list-windows -t "${SESSION}" -F '#W' | rg -Fxq preflight; then
  command=(
    "${PYTHON}"
    "${SCRIPT_DIR}/run_s_depth_strata_preflight.py"
    --config "${CONFIG}"
  )
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-window -d -t "${SESSION}" -n preflight \
    "${shell_command}; exec bash"
fi

if ! tmux list-windows -t "${SESSION}" -F '#W' | rg -Fxq coordinator; then
  command=(
    "${PYTHON}"
    "${SCRIPT_DIR}/run_s_depth_strata_coordinator.py"
    --config "${CONFIG}"
  )
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-window -d -t "${SESSION}" -n coordinator \
    "${shell_command}; exec bash"
fi

for gpu in "${GPUS[@]}"; do
  name="sdepth_g${gpu}"
  if tmux list-windows -t "${SESSION}" -F '#W' | rg -Fxq "${name}"; then
    echo "window already exists: ${SESSION}:${name}"
    continue
  fi
  command=(
    "${PYTHON}"
    "${SCRIPT_DIR}/run_s_depth_strata_worker.py"
    --config "${CONFIG}"
    --gpu "${gpu}"
    --worker-id "${name}"
  )
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-window -d -t "${SESSION}" -n "${name}" \
    "${shell_command}; exec bash"
done

tmux select-window -t "${SESSION}:preflight"
tmux list-windows -t "${SESSION}"
