#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VJEPA2_ROOT="${VJEPA2_ROOT:-/home/gaoya/Code_Video/vjepa2-main}"
VJEPA2_CHECKPOINT="${VJEPA2_CHECKPOINT:-/data/gaoya/agent-data/weights/vjepa2_1_vitl_dist_vitG_384_ema_encoder.pt}"
DATA_DIR="${DATA_DIR:-/data/gaoya/dataset}"
SAVE_DIR="${SAVE_DIR:-/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_prefix_causal_movi_c}"
YTVIS_CHECKPOINT="${YTVIS_CHECKPOINT:-}"
GPU_IDS="${GPU_IDS:-5,6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-xssc_vjepa2_1_video}"
WANDB_MODE="${WANDB_MODE:-online}"
MAX_STEP="${MAX_STEP:-}"
RESUME_FILE="${RESUME_FILE:-}"

if [[ ! -f "${VJEPA2_ROOT}/src/hub/backbones.py" ]]; then
  echo "ERROR: V-JEPA2 repository is missing: ${VJEPA2_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${VJEPA2_CHECKPOINT}" ]]; then
  echo "ERROR: V-JEPA2.1 checkpoint is missing: ${VJEPA2_CHECKPOINT}" >&2
  exit 2
fi
if [[ -z "${RESUME_FILE}" && ! -f "${YTVIS_CHECKPOINT}" ]]; then
  echo "ERROR: causal YTVIS xSSC checkpoint is missing: ${YTVIS_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${SAVE_DIR}"
cd "${ROOT}"
extra_args=()
if [[ -n "${MAX_STEP}" ]]; then
  extra_args+=(--max-step "${MAX_STEP}")
fi
if [[ -n "${RESUME_FILE}" ]]; then
  extra_args+=(--resume-file "${RESUME_FILE}")
else
  extra_args+=(--ckpt-file "${YTVIS_CHECKPOINT}")
fi
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
  --cfg-file upstream/config-randsfq/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-slot512-prefix-causal-transfer15000.py \
  --data-dir "${DATA_DIR}" \
  --save-dir "${SAVE_DIR}" \
  "${extra_args[@]}"
