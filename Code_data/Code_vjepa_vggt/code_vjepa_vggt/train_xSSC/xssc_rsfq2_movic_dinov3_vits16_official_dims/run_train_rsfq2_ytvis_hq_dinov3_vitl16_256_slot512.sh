#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DINOV3_ROOT="${DINOV3_ROOT:-${ROOT}/third_party/dinov3}"
DINOV3_CHECKPOINT="${DINOV3_CHECKPOINT:-/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the official converted xSSC dataset root}"
SAVE_DIR="${SAVE_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:?Set WANDB_PROJECT for formal training tracking}"
WANDB_MODE="${WANDB_MODE:-online}"

for split in train val; do
  path="${DATA_DIR}/ytvis_hq/${split}.lmdb"
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
  -m torch.distributed.run
  --standalone
  --nproc-per-node="${NPROC_PER_NODE}"
  train_ddp_ytvis_hq.py
  --seed "${SEED}"
  --cfg-file upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py
  --data-dir "${DATA_DIR}"
  --save-dir "${SAVE_DIR}"
)
cmd+=(--project "${WANDB_PROJECT}")

echo "[xssc-dinov3] xSSC=90a0ef1c3cc02c05e7a6abcee7b1adeaca107967"
echo "[xssc-dinov3] DINOv3=6876159a11b4df116f30f667f8c9888617df0751"
echo "[xssc-dinov3] input=256x256 grid=16x16 feature_dim=1024 slot_dim=512 seed=${SEED} gpus=${GPU_IDS}"
echo "[xssc-dinov3] dataset=YTVIS-HQ data=${DATA_DIR} save=${SAVE_DIR}"

cd "${ROOT}"
exec env \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  DINOV3_ROOT="${DINOV3_ROOT}" \
  DINOV3_CHECKPOINT="${DINOV3_CHECKPOINT}" \
  WANDB_MODE="${WANDB_MODE}" \
  PYTHONPATH="${ROOT}/upstream:${DINOV3_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${cmd[@]}"
