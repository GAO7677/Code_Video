#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/data/gaoya/dataset}"
SAVE_DIR="${SAVE_DIR:-/data/gaoya/agent-data/checkpoints/xssc_stage1_causal_state_from25000_gpu0}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_noncausal_movi_c_10f_transfer16000_clip2_steps50000/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-10f-slot512-transfer16000-clip2/42/step-025000.pth}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
STAGE1_BATCH_SIZE_T="${STAGE1_BATCH_SIZE_T:?Set STAGE1_BATCH_SIZE_T after the capacity probe}"
WANDB_PROJECT="${WANDB_PROJECT:-xssc_stage1_causal_state_from25000}"
WANDB_MODE="${WANDB_MODE:-online}"
CONFIG="upstream/config-randsfq/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-prefix-causal-from25000-gpu0.py"

if [[ "${GPU_ID}" != "0" ]]; then
  echo "ERROR: this branch is intentionally pinned to physical GPU 0" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "ERROR: source checkpoint is missing: ${SOURCE_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT%.pth}.metadata.json" ]]; then
  echo "ERROR: source checkpoint metadata is missing" >&2
  exit 2
fi
if (( 384 % STAGE1_BATCH_SIZE_T != 0 )); then
  echo "ERROR: STAGE1_BATCH_SIZE_T must divide effective batch 384" >&2
  exit 2
fi

mkdir -p "${SAVE_DIR}"
cd "${ROOT}"
exec env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  STAGE1_BATCH_SIZE_T="${STAGE1_BATCH_SIZE_T}" \
  WANDB_MODE="${WANDB_MODE}" \
  PYTHONPATH="${ROOT}/upstream:/home/gaoya/Code_Video/vjepa2-main${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nproc-per-node=1 \
  train_ddp_ytvis_hq.py \
  --project "${WANDB_PROJECT}" \
  --seed "${SEED}" \
  --cfg-file "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --ckpt-file "${SOURCE_CHECKPOINT}"
