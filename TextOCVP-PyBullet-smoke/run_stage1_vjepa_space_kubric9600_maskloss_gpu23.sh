#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/checkpoints/savi_vjepa_space_kubric9600_maskloss_gpu23_${RUN_TAG}}"
WANDB_GROUP="${WANDB_GROUP:-savi_pixel_vs_vjepa_kubric9600_maskloss_${RUN_TAG}}"

GPU_IDS=2,3 \
DATASET_MODE=kubric \
MASTER_PORT="${MASTER_PORT:-29652}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
WANDB_GROUP="${WANDB_GROUP}" \
PER_GPU_BATCH_SIZE=16 \
EFFECTIVE_BATCH_SIZE=64 \
bash /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/run_stage1_vjepa_space.sh \
  --index-root /data/gaoya/agent-data/datasets/savi_indices_kubric9600 \
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
  --wandb-name "savi_vjepa_space_kubric9600_maskloss_${RUN_TAG}"
