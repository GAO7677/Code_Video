#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
PROJECT=/home/gaoya/Code_Video/TextOCVP-PyBullet-smoke
INDEX_ROOT=/data/gaoya/agent-data/datasets/savi_indices_kubric9600
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ROOT:-/data/gaoya/agent-data/checkpoints/savi_pixel_collapse_diagnosis_gpu56_${RUN_TAG}}"
WANDB_GROUP="${WANDB_GROUP:-savi_pixel_collapse_diagnosis_${RUN_TAG}}"
MAX_STEPS="${MAX_STEPS:-4000}"

mkdir -p "${ROOT}"
printf 'RUN_TAG=%s\nROOT=%s\nWANDB_GROUP=%s\n' \
  "${RUN_TAG}" "${ROOT}" "${WANDB_GROUP}" > "${ROOT}/run_manifest.env"

run_control() {
  local name="$1"
  local height="$2"
  local width="$3"
  local per_gpu_batch="$4"
  local output_dir="${ROOT}/${name}"

  echo "[$(date -u +%FT%TZ)] starting ${name}: ${height}x${width}"
  "${PYTHON}" "${PROJECT}/launch_stage1_experiment.py" \
    --dataset-mode kubric \
    --index-root "${INDEX_ROOT}" \
    --output-dir "${output_dir}" \
    --gpus 5,6 \
    --distributed \
    --per-gpu-batch-size "${per_gpu_batch}" \
    --effective-batch-size 64 \
    --mixed-precision bf16 \
    --master-port 29661 \
    --image-height "${height}" \
    --image-width "${width}" \
    --num-slots 8 \
    --slot-dim 256 \
    --epochs 1000 \
    --max-optimizer-steps "${MAX_STEPS}" \
    --warmup-steps 2000 \
    --validation-frequency-steps 500 \
    --mask-loss-weight 0 \
    --wandb-project textocvp_savi_stage1 \
    --wandb-group "${WANDB_GROUP}"
  echo "[$(date -u +%FT%TZ)] completed ${name}"
}

run_control phase1a_lowres64_pure_mse 64 64 32
run_control phase1b_highres216x384_pure_mse 216 384 4

echo "[$(date -u +%FT%TZ)] Phase 1 controls completed"
