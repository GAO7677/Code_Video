#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VJEPA2_ROOT="${VJEPA2_ROOT:-/home/gaoya/Code_Video/vjepa2-main}"
VJEPA2_CHECKPOINT="${VJEPA2_CHECKPOINT:-/data/gaoya/agent-data/weights/vjepa2_1_vitl_dist_vitG_384_ema_encoder.pt}"
DATA_DIR="${DATA_DIR:-/data/gaoya/dataset}"
SAVE_DIR="${SAVE_DIR:-/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_noncausal_movi_c_10f_transfer16000_clip2_steps50000}"
YTVIS_CHECKPOINT="${YTVIS_CHECKPOINT:-/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_noncausal_ytvis_hq_10f_ar_bs64_steps20000_clip2_resume14000/rsfq2_r-ytvis_hq-vjepa2_1_vitl16-ar10f-slot512-resume14000-clip2-bs64/42/step-016000.pth}"
GPU_IDS="${GPU_IDS:-5,6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-xssc_vjepa2_1_movi_c_10f_clip2}"
WANDB_MODE="${WANDB_MODE:-online}"
MAX_STEP="${MAX_STEP:-}"
RESUME_FILE="${RESUME_FILE:-}"
CONFIG="upstream/config-randsfq/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-10f-slot512-transfer16000-clip2.py"

if [[ ! -f "${VJEPA2_ROOT}/src/hub/backbones.py" ]]; then
  echo "ERROR: V-JEPA2 repository is missing: ${VJEPA2_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${VJEPA2_CHECKPOINT}" ]]; then
  echo "ERROR: V-JEPA2.1 checkpoint is missing: ${VJEPA2_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -d "${DATA_DIR}/kubric-movi/movi-c/1.0.0" ]]; then
  echo "ERROR: MOVi-C dataset is missing below ${DATA_DIR}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

extra_args=()
if [[ -n "${MAX_STEP}" ]]; then
  extra_args+=(--max-step "${MAX_STEP}")
fi
if [[ -n "${RESUME_FILE}" ]]; then
  if [[ ! -f "${RESUME_FILE}" ]]; then
    echo "ERROR: resume state is missing: ${RESUME_FILE}" >&2
    exit 2
  fi
  extra_args+=(--resume-file "${RESUME_FILE}")
else
  if [[ ! -f "${YTVIS_CHECKPOINT}" || ! -f "${YTVIS_CHECKPOINT%.pth}.metadata.json" ]]; then
    echo "ERROR: complete YTVIS step-16000 checkpoint is missing: ${YTVIS_CHECKPOINT}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" - "${YTVIS_CHECKPOINT%.pth}.metadata.json" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
expected_variant = (
    "vjepa2_1_vitl16_video_ytvis_hq_10f_ar_slot512_transfer10000_bs64"
)
if metadata.get("optimizer_step") != 16000:
    raise SystemExit(f"unexpected source step: {metadata}")
if metadata.get("variant_name") != expected_variant:
    raise SystemExit(f"unexpected source variant: {metadata}")
if metadata.get("world_size") != 2:
    raise SystemExit(f"unexpected source world size: {metadata}")
if metadata.get("effective_global_batch_size") != 384:
    raise SystemExit(f"unexpected source effective batch: {metadata}")
print(f"[movi-launch] verified source metadata: {metadata}", flush=True)
PY
  extra_args+=(--ckpt-file "${YTVIS_CHECKPOINT}")
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
  --cfg-file "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  --save-dir "${SAVE_DIR}" \
  "${extra_args[@]}"
