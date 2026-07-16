#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
EPOCHS="${EPOCHS:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623_savi/experiments/vjepa_space_pybullet1_kubric8_gpu56_${RUN_TAG}}"

GPU_IDS=5,6 \
DATASET_MODE=mixed \
MASTER_PORT="${MASTER_PORT:-29638}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
WANDB_GROUP="vjepa_space_pybullet1_kubric8_${RUN_TAG}" \
bash /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/run_stage1_vjepa_space.sh \
  --index-root /data/gaoya/AAA_test_video/0623_savi/indices_pybullet1200_kubric9600_full_pool \
  --source-sampling-ratio 1:8 \
  --samples-per-epoch 10800 \
  --per-gpu-batch-size 1 \
  --effective-batch-size 16 \
  --epochs "${EPOCHS}" \
  --validation-frequency-steps 500 \
  "$@"

