#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="${ROOT}/upstream"
DINOV3_ROOT="${DINOV3_ROOT:-${ROOT}/third_party/dinov3}"
DINOV3_CHECKPOINT="${DINOV3_CHECKPOINT:-/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the official converted xSSC dataset root}"
SAVE_DIR="${SAVE_DIR:-/data/gaoya/agent-data/checkpoints/xssc_rsfq2_ytvis_dinov3_vitl16_256}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

for split in train val; do
  path="${DATA_DIR}/ytvis_2022/${split}.lmdb"
  if [ ! -e "${path}" ]; then
    echo "ERROR: required official converted dataset is missing: ${path}" >&2
    exit 2
  fi
done
if [ ! -f "${DINOV3_CHECKPOINT}" ]; then
  echo "ERROR: DINOv3 checkpoint is missing: ${DINOV3_CHECKPOINT}" >&2
  exit 2
fi
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${SAVE_DIR}"
cmd=(
  "${PYTHON_BIN}"
  train.py
  --seed "${SEED}"
  --cfg_file config-randsfq/rsfq2_r-ytvis.py
  --data_dir "${DATA_DIR}"
  --save_dir "${SAVE_DIR}"
)
if [ -n "${WANDB_PROJECT}" ]; then
  cmd+=(--project "${WANDB_PROJECT}")
fi

echo "[xssc-dinov3] xSSC=90a0ef1c3cc02c05e7a6abcee7b1adeaca107967"
echo "[xssc-dinov3] DINOv3=6876159a11b4df116f30f667f8c9888617df0751"
echo "[xssc-dinov3] input=256x256 grid=16x16 feature_dim=1024 seed=${SEED} gpu=${GPU_ID}"
echo "[xssc-dinov3] data=${DATA_DIR} save=${SAVE_DIR}"

cd "${UPSTREAM}"
exec env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  DINOV3_ROOT="${DINOV3_ROOT}" \
  DINOV3_CHECKPOINT="${DINOV3_CHECKPOINT}" \
  PYTHONPATH="${UPSTREAM}:${DINOV3_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${cmd[@]}"
