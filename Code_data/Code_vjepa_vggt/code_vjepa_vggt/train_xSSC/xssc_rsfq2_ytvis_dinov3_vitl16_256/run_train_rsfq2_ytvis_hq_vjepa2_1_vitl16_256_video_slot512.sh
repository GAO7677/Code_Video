#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VJEPA2_ROOT="${VJEPA2_ROOT:-/home/gaoya/Code_Video/vjepa2-main}"
VJEPA2_CHECKPOINT="${VJEPA2_CHECKPOINT:-/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt}"
DATA_DIR="${DATA_DIR:-/data/gaoya/dataset}"
SAVE_DIR="${SAVE_DIR:-/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_ytvis_hq}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-xssc_vjepa2_1_video}"
WANDB_MODE="${WANDB_MODE:-online}"

for split in train val; do
  path="${DATA_DIR}/ytvis_hq/${split}.lmdb"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required YTVIS-HQ dataset is missing: ${path}" >&2
    exit 2
  fi
done
if [[ ! -f "${VJEPA2_ROOT}/src/hub/backbones.py" ]]; then
  echo "ERROR: V-JEPA2 repository is missing: ${VJEPA2_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${VJEPA2_CHECKPOINT}" ]]; then
  echo "ERROR: V-JEPA2.1 checkpoint is missing: ${VJEPA2_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${SAVE_DIR}"
cd "${ROOT}"
exec env \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  VJEPA2_ROOT="${VJEPA2_ROOT}" \
  VJEPA2_CHECKPOINT="${VJEPA2_CHECKPOINT}" \
  WANDB_MODE="${WANDB_MODE}" \
  PYTHONPATH="${ROOT}/upstream:${VJEPA2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${NPROC_PER_NODE}" \
  train_ddp_ytvis_hq.py \
  --project "${WANDB_PROJECT}" \
  --seed "${SEED}" \
  --cfg-file upstream/config-randsfq/rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512.py \
  --data-dir "${DATA_DIR}" \
  --save-dir "${SAVE_DIR}"
