#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_head_role_dose_control_preflight_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION=wan_head_role_dose_control
INPUT_LIST="${SCRIPT_DIR}/common22_public_head_ablation_preflight_case.txt"
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/preflight_generation
MANIFEST=/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/configs/matched_subsets.json
SUBSET_ID=S_k08_r00_depthmatch

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n shell
fi

models=(wan_lora xssc physrvg)
gpus=(0 1 2)
for index in "${!models[@]}"; do
  model="${models[index]}"
  gpu="${gpus[index]}"
  name="pre_${model}"
  if tmux list-windows -t "${SESSION}" -F '#W' | rg -Fxq "${name}"; then
    tmux kill-window -t "${SESSION}:${name}"
  fi
  command=(
    env
    "MODEL=${model}"
    SEED=851
    "SUBSET_ID=${SUBSET_ID}"
    "GPU=${gpu}"
    STEP_START=0
    STEP_END=10
    "INPUT_LIST=${INPUT_LIST}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    "MANIFEST=${MANIFEST}"
    bash
    "${SCRIPT_DIR}/run_matched_head_subset_ablation_job.sh"
  )
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-window -d -t "${SESSION}" -n "${name}" \
    "${shell_command}; exec bash"
done

tmux select-window -t "${SESSION}:pre_wan_lora"
tmux list-windows -t "${SESSION}"
