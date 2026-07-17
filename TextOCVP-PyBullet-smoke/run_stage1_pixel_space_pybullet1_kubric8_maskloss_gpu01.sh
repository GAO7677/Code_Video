#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
PROJECT=/home/gaoya/Code_Video/TextOCVP-PyBullet-smoke
INDEX_ROOT=/data/gaoya/AAA_test_video/0623_savi/indices_pybullet1200_kubric9600_full_pool
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_IDS="${GPU_IDS:-0,1}"
MICRO_GLOBAL_BATCH_SIZE="${MICRO_GLOBAL_BATCH_SIZE:-8}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-1000}"
VALIDATION_FREQUENCY_STEPS="${VALIDATION_FREQUENCY_STEPS:-500}"
MASK_LOSS_WEIGHT="${MASK_LOSS_WEIGHT:-1.0}"
MASK_LOSS_WARMUP_STEPS="${MASK_LOSS_WARMUP_STEPS:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/checkpoints/savi_pixel_space_pybullet1_kubric8_maskloss_gpu01_${RUN_TAG}}"
WANDB_GROUP="${WANDB_GROUP:-savi_pixel_space_pybullet1_kubric8_maskloss_${RUN_TAG}}"

if (( EFFECTIVE_BATCH_SIZE % MICRO_GLOBAL_BATCH_SIZE != 0 )); then
  echo "EFFECTIVE_BATCH_SIZE must be divisible by MICRO_GLOBAL_BATCH_SIZE" >&2
  exit 2
fi

args=(
  --dataset-mode mixed
  --index-root "${INDEX_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --gpus "${GPU_IDS}"
  --micro-global-batch-size "${MICRO_GLOBAL_BATCH_SIZE}"
  --effective-batch-size "${EFFECTIVE_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --validation-frequency-steps "${VALIDATION_FREQUENCY_STEPS}"
  --mask-loss-weight "${MASK_LOSS_WEIGHT}"
  --mask-loss-warmup-steps "${MASK_LOSS_WARMUP_STEPS}"
  --mask-max-instances 6
  --mask-union-weight 0.20
  --mask-instance-weight 0.10
  --mask-static-weight 0.02
  --mask-background-weight 0.01
  --mask-unused-weight 0.01
  --mask-focal-bce-weight 0.25
  --wandb-project textocvp_savi_stage1
  --wandb-group "${WANDB_GROUP}"
)

if [[ -n "${MAX_OPTIMIZER_STEPS:-}" ]]; then
  args+=(--max-optimizer-steps "${MAX_OPTIMIZER_STEPS}")
fi
if [[ "${DISABLE_WANDB:-0}" == "1" ]]; then
  args+=(--disable-wandb)
fi

"${PYTHON}" "${PROJECT}/launch_stage1_experiment.py" "${args[@]}"
