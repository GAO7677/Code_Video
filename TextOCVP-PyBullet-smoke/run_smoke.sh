#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PROJECT=/home/gaoya/Code_Video/TextOCVP-PyBullet-smoke
DATASET_ROOT=/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/checkpoints/textocvp_pybullet_smoke_${RUN_TAG}}"
HF_HOME="${HF_HOME:-/data/gaoya/agent-data/cache/huggingface}"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

env \
  PYTHONNOUSERSITE=1 \
  HF_HOME="${HF_HOME}" \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON}" "${PROJECT}/train_smoke.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --gpu "${GPU_ID:-6}" \
    --dataset-limit "${DATASET_LIMIT:-32}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --decomp-steps "${DECOMP_STEPS:-6}" \
    --predictor-steps "${PREDICTOR_STEPS:-3}" \
    --height "${HEIGHT:-64}" \
    --width "${WIDTH:-112}" \
    --num-frames 10 \
    --num-slots "${NUM_SLOTS:-6}" \
    --slot-dim "${SLOT_DIM:-128}" \
  2>&1 | tee "${OUTPUT_DIR}/smoke.log"

echo "smoke output: ${OUTPUT_DIR}"

