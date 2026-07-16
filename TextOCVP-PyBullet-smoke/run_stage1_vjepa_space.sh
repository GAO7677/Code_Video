#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATASET_MODE="${DATASET_MODE:-mixed}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623_savi/experiments/vjepa_space_stage1_${DATASET_MODE}_${RUN_TAG}}"
WANDB_GROUP="${WANDB_GROUP:-feature_space_stage1_${RUN_TAG}}"
MASTER_PORT="${MASTER_PORT:-29631}"
NPROC="$(awk -F, '{print NF}' <<<"${GPU_IDS}")"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-48}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-$((PER_GPU_BATCH_SIZE * NPROC))}"

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONNOUSERSITE=1

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/train_stage1_vjepa_space.py \
  --checkpoint /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth \
  --output-dir "${OUTPUT_DIR}" \
  --index-root /data/gaoya/AAA_test_video/0623_savi/indices \
  --dataset-mode "${DATASET_MODE}" \
  --dataset-preprocess-mode vjepa \
  --num-frames 10 \
  --image-height 384 \
  --image-width 384 \
  --num-slots 8 \
  --slot-dim 512 \
  --per-gpu-batch-size "${PER_GPU_BATCH_SIZE}" \
  --effective-batch-size "${EFFECTIVE_BATCH_SIZE}" \
  --epochs 1000 \
  --validation-frequency-steps 500 \
  --wandb-group "${WANDB_GROUP}" \
  "$@"
