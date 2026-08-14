#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 WAIT_WORKER TARGET_WORKER GPU_ID" >&2
  exit 2
fi
WAIT_WORKER="$1"
TARGET_WORKER="$2"
GPU_ID="$3"
if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU4 is prohibited by workspace rules" >&2
  exit 2
fi

OUTPUT_ROOT=/data/gaoya/agent-data/outputs/train_subset_val_loss_seed42
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/xssc_loss_project/evaluate_train_subset_val_loss.py

while pgrep -f "evaluate_train_subset_val_loss.py.*--worker-id ${WAIT_WORKER}" >/dev/null; do
  sleep 60
done

exec env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" -u "${SCRIPT}" \
  --output-root "${OUTPUT_ROOT}" \
  --worker-id "${TARGET_WORKER}" \
  --num-workers 4 \
  --repeat-check
