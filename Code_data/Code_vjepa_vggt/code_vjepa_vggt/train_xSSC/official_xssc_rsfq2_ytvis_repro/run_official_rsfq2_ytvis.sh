#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="${ROOT}/upstream"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the converted xSSC dataset root}"
SAVE_DIR="${SAVE_DIR:-/data/gaoya/agent-data/checkpoints/xssc_official_reproduction}"
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

echo "[official-xssc] commit=90a0ef1c3cc02c05e7a6abcee7b1adeaca107967"
echo "[official-xssc] config=config-randsfq/rsfq2_r-ytvis.py seed=${SEED} gpu=${GPU_ID}"
echo "[official-xssc] data=${DATA_DIR} save=${SAVE_DIR}"

cd "${UPSTREAM}"
exec env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH="${UPSTREAM}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${cmd[@]}"
