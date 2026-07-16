#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATASET_MODE="${DATASET_MODE:-mixed}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623_savi/experiments/vae_space_stage1_${DATASET_MODE}_${RUN_TAG}}"
WANDB_GROUP="${WANDB_GROUP:-feature_space_stage1_${RUN_TAG}}"
MASTER_PORT="${MASTER_PORT:-29632}"
NPROC="$(awk -F, '{print NF}' <<<"${GPU_IDS}")"

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONNOUSERSITE=1

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  /home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/train_stage1_vae_space.py \
  --checkpoint /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --output-dir "${OUTPUT_DIR}" \
  --index-root /data/gaoya/AAA_test_video/0623_savi/indices \
  --dataset-mode "${DATASET_MODE}" \
  --num-frames 9 \
  --image-height 216 \
  --image-width 384 \
  --num-slots 8 \
  --slot-dim 256 \
  --per-gpu-batch-size 2 \
  --effective-batch-size 16 \
  --epochs 1000 \
  --validation-frequency-steps 500 \
  --wandb-group "${WANDB_GROUP}" \
  "$@"

