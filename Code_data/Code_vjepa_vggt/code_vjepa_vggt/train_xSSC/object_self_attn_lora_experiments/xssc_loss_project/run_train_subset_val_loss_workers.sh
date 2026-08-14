#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-train_subset_val_loss_seed42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/train_subset_val_loss_seed42}"
GPU_SET="${GPU_SET:-0,3,5,6}"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/xssc_loss_project/evaluate_train_subset_val_loss.py
MONITOR_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/xssc_loss_project/monitor_train_subset_val_loss_report.sh

IFS=',' read -r -a GPUS <<<"${GPU_SET}"
if [[ " ${GPUS[*]} " == *" 4 "* ]]; then
  echo "GPU4 is prohibited by workspace rules" >&2
  exit 2
fi
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_SET must contain exactly four GPU IDs" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"
worker_command() {
  local worker="$1"
  local gpu="$2"
  printf '%q ' env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON}" -u "${SCRIPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --worker-id "${worker}" \
    --num-workers 4 \
    --repeat-check
  printf '2>&1 | tee -a %q' "${OUTPUT_ROOT}/logs/worker-${worker}-gpu${gpu}.log"
}

tmux new-session -d -s "${SESSION}" -n "worker0-gpu${GPUS[0]}" \
  "bash -lc 'set -o pipefail; $(worker_command 0 "${GPUS[0]}")'"
for worker in 1 2 3; do
  tmux new-window -t "${SESSION}" -n "worker${worker}-gpu${GPUS[$worker]}" \
    "bash -lc 'set -o pipefail; $(worker_command "${worker}" "${GPUS[$worker]}")'"
done

tmux new-window -t "${SESSION}" -n report \
  "bash -lc '${MONITOR_SCRIPT} >>${OUTPUT_ROOT}/logs/report-builder.log 2>&1'"

echo "session=${SESSION}"
echo "gpu_set=${GPU_SET}"
echo "output_root=${OUTPUT_ROOT}"
tmux list-windows -t "${SESSION}" -F '#I:#W pane_pid=#{pane_pid}'
