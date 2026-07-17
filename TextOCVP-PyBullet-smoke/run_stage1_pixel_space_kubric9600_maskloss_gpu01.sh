#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
PROJECT=/home/gaoya/Code_Video/TextOCVP-PyBullet-smoke
INDEX_ROOT=/data/gaoya/agent-data/datasets/savi_indices_kubric9600
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/checkpoints/savi_pixel_space_kubric9600_maskloss_gpu01_${RUN_TAG}}"
WANDB_GROUP="${WANDB_GROUP:-savi_pixel_vs_vjepa_kubric9600_maskloss_${RUN_TAG}}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" "${PROJECT}/launch_stage1_experiment.py" \
  --dataset-mode kubric \
  --index-root "${INDEX_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --gpus 0,1 \
  --distributed \
  --per-gpu-batch-size 4 \
  --effective-batch-size 64 \
  --mixed-precision bf16 \
  --master-port "${MASTER_PORT:-29651}" \
  --epochs "${EPOCHS:-1000}" \
  --max-optimizer-steps "${MAX_OPTIMIZER_STEPS:-9000}" \
  --validation-frequency-steps "${VALIDATION_FREQUENCY_STEPS:-500}" \
  --mask-loss-weight "${MASK_LOSS_WEIGHT:-1.0}" \
  --mask-loss-warmup-steps "${MASK_LOSS_WARMUP_STEPS:-500}" \
  --mask-max-instances 6 \
  --mask-union-weight 0.20 \
  --mask-instance-weight 0.10 \
  --mask-static-weight 0.02 \
  --mask-background-weight 0.01 \
  --mask-unused-weight 0.01 \
  --mask-focal-bce-weight 0.25 \
  --wandb-project textocvp_savi_stage1 \
  --wandb-group "${WANDB_GROUP}"
