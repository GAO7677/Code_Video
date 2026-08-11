#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT}/upstream/config-randsfq/rsfq2_r_recogn-ytvis_hq-vjepa2_1_vitl16_256-video-slot512-step7000-pilot.py}"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: config file is missing: ${CONFIG_FILE}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable is missing: ${PYTHON_BIN}" >&2
  exit 2
fi

mapfile -t SETTINGS < <(
  env PYTHONPATH="${ROOT}/upstream${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - "${CONFIG_FILE}" <<'PY'
from pathlib import Path
import sys

from object_centric_bench.util import Config

cfg = Config.fromfile(Path(sys.argv[1]).resolve())
values = [
    ",".join(str(gpu) for gpu in cfg.gpu_ids),
    str(cfg.expected_world_size),
    str(cfg.seed),
    str(cfg.data_dir),
    str(cfg.save_dir),
    str(cfg.wandb_project),
    str(cfg.wandb_mode),
    str(cfg.source_checkpoint),
]
print("\n".join(values))
PY
)

if [[ "${#SETTINGS[@]}" -ne 8 ]]; then
  echo "ERROR: failed to read runtime settings from ${CONFIG_FILE}" >&2
  exit 2
fi

GPU_IDS="${SETTINGS[0]}"
NPROC_PER_NODE="${SETTINGS[1]}"
SEED="${SETTINGS[2]}"
DATA_DIR="${SETTINGS[3]}"
SAVE_DIR="${SETTINGS[4]}"
WANDB_PROJECT="${SETTINGS[5]}"
WANDB_MODE="${SETTINGS[6]}"
SOURCE_CHECKPOINT="${SETTINGS[7]}"

if [[ ",${GPU_IDS}," == *",4,"* ]]; then
  echo "ERROR: GPU 4 is forbidden by workspace policy" >&2
  exit 2
fi
for split in train val; do
  if [[ ! -f "${DATA_DIR}/ytvis_hq/${split}.lmdb" ]]; then
    echo "ERROR: missing YTVIS-HQ ${split} LMDB: ${DATA_DIR}/ytvis_hq/${split}.lmdb" >&2
    exit 2
  fi
done
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "ERROR: source xSSC checkpoint is missing: ${SOURCE_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${SAVE_DIR}"
cd "${ROOT}"
exec env \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_MODE="${WANDB_MODE}" \
  PYTHONPATH="${ROOT}/upstream${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${NPROC_PER_NODE}" \
  train_ddp_ytvis_hq.py \
  --project "${WANDB_PROJECT}" \
  --seed "${SEED}" \
  --cfg-file "${CONFIG_FILE}" \
  --data-dir "${DATA_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --ckpt-file "${SOURCE_CHECKPOINT}"
