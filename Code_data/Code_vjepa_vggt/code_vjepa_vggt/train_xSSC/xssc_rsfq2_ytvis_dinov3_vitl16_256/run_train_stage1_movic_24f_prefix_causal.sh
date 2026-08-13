#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/data/gaoya/dataset}"
SAVE_DIR="${SAVE_DIR:-/data/gaoya/agent-data/checkpoints/xssc_stage1_causal_state}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT to the selected completed MOVi-C step-050000 checkpoint}"
GPU_IDS="${GPU_IDS:?Set GPU_IDS to two available GPUs; GPU 4 is prohibited}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-xssc_stage1_causal_state}"
WANDB_MODE="${WANDB_MODE:-online}"
CONFIG="upstream/config-randsfq/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-prefix-causal-stage1.py"

IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
if [[ "${#gpu_array[@]}" -ne 2 ]]; then
  echo "ERROR: Stage-1 causal adaptation requires exactly two GPUs" >&2
  exit 2
fi
for gpu_id in "${gpu_array[@]}"; do
  if [[ "${gpu_id}" == "4" ]]; then
    echo "ERROR: GPU 4 is prohibited by workspace policy" >&2
    exit 2
  fi
done
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "ERROR: source checkpoint is missing: ${SOURCE_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT%.pth}.metadata.json" ]]; then
  echo "ERROR: source checkpoint metadata is missing" >&2
  exit 2
fi

mkdir -p "${SAVE_DIR}"
cd "${ROOT}"
exec env \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_MODE="${WANDB_MODE}" \
  PYTHONPATH="${ROOT}/upstream:/home/gaoya/Code_Video/vjepa2-main${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  train_ddp_ytvis_hq.py \
  --project "${WANDB_PROJECT}" \
  --seed "${SEED}" \
  --cfg-file "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --ckpt-file "${SOURCE_CHECKPOINT}"

