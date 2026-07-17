#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
EPOCHS="${EPOCHS:-1000}"
MASK_LOSS_WEIGHT="${MASK_LOSS_WEIGHT:-1.0}"
MASK_LOSS_WARMUP_STEPS="${MASK_LOSS_WARMUP_STEPS:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623_savi/experiments/vjepa_space_pybullet1_kubric8_maskloss_gpu56_${RUN_TAG}}"

GPU_IDS=5,6 \
DATASET_MODE=mixed \
MASTER_PORT="${MASTER_PORT:-29639}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
WANDB_GROUP="vjepa_space_pybullet1_kubric8_maskloss_${RUN_TAG}" \
PER_GPU_BATCH_SIZE=48 \
EFFECTIVE_BATCH_SIZE=96 \
bash /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/run_stage1_vjepa_space.sh \
  --index-root /data/gaoya/AAA_test_video/0623_savi/indices_pybullet1200_kubric9600_full_pool \
  --source-sampling-ratio 1:8 \
  --samples-per-epoch 10800 \
  --epochs "${EPOCHS}" \
  --validation-frequency-steps 500 \
  --mask-loss-weight "${MASK_LOSS_WEIGHT}" \
  --mask-loss-warmup-steps "${MASK_LOSS_WARMUP_STEPS}" \
  --mask-max-instances 6 \
  --mask-union-weight 0.20 \
  --mask-instance-weight 0.10 \
  --mask-static-weight 0.02 \
  --mask-background-weight 0.01 \
  --mask-unused-weight 0.01 \
  --mask-focal-bce-weight 0.25 \
  "$@"
