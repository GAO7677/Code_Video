#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DATA_DIR="${DATA_DIR:-/data/gaoya/dataset}"
DINOV3_ROOT="${DINOV3_ROOT:-${ROOT}/third_party/dinov3}"
DINOV3_CHECKPOINT="${DINOV3_CHECKPOINT:-/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-/data/gaoya/agent-data/checkpoints/xssc_slot512_ddp_smoke/slot512_smoke_nonbackbone.pth}"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"

cd "${ROOT}"
exec env \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  DINOV3_ROOT="${DINOV3_ROOT}" \
  DINOV3_CHECKPOINT="${DINOV3_CHECKPOINT}" \
  PYTHONPATH="${ROOT}/upstream:${DINOV3_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${NPROC_PER_NODE}" \
  smoke_train_ddp_ytvis_hq.py \
  --data-dir "${DATA_DIR}" \
  --checkpoint-file "${CHECKPOINT_FILE}" \
  --steps 1 \
  --amp-dtype bfloat16
