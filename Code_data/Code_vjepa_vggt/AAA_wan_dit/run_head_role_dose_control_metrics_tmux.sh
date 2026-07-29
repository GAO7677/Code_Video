#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_head_role_dose_control_metrics_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${SCRIPT_DIR}/head_role_dose_control_pilot.json"
SESSION=wan_head_role_dose_control
ROOT=/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/pilot
READY="${ROOT}/metrics.ready"
INPUT_LIST=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/input_lists/test5_unique20.txt
WAIT_WORKER="${SCRIPT_DIR}/run_test5_ablation_metric_wait_worker.sh"
GPUS=(1 2 3 5 6 7)

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n shell
fi

if ! tmux list-windows -t "${SESSION}" -F '#W' | rg -Fxq coordinator; then
  command=(
    "${PYTHON}"
    "${SCRIPT_DIR}/run_head_role_dose_control_coordinator.py"
    --config "${CONFIG}"
  )
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-window -d -t "${SESSION}" -n coordinator \
    "${shell_command}; exec bash"
fi

for gpu in "${GPUS[@]}"; do
  for kind_count in cpu:2 gpu_common:1 videophy2:1 cosmos:1; do
    kind="${kind_count%%:*}"
    count="${kind_count##*:}"
    for ((index=0; index<count; index++)); do
      name="m_g${gpu}_${kind}_${index}"
      if tmux list-windows -t "${SESSION}" -F '#W' | rg -Fxq "${name}"; then
        continue
      fi
      command=(
        bash
        "${WAIT_WORKER}"
        "${gpu}"
        "${kind}"
        "${name}"
        "${ROOT}"
        "${INPUT_LIST}"
        "${READY}"
      )
      printf -v shell_command '%q ' "${command[@]}"
      tmux new-window -d -t "${SESSION}" -n "${name}" \
        "${shell_command}; exec bash"
    done
  done
done

tmux select-window -t "${SESSION}:coordinator"
tmux list-windows -t "${SESSION}"
