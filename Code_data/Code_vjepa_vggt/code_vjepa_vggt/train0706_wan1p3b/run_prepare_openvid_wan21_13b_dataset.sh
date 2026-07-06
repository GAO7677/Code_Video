#!/usr/bin/env bash
# Build a Wan2.1-1.3B-ready OpenVid parquet dataset root from newly downloaded shards.
# Run:
#   sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_prepare_openvid_wan21_13b_dataset.sh
#
# Optional override:
#   INPUT_ROOT=/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train \
#   OUTPUT_ROOT=/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train_wan21_13b_ready_ctx24 \
#   sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_prepare_openvid_wan21_13b_dataset.sh
set -euo pipefail

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT=${PROJ}/code_vjepa_vggt/train0706_wan1p3b/prepare_openvid_wan21_13b_dataset.py

INPUT_ROOT=${INPUT_ROOT:-/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train_wan21_13b_ready_ctx24}
NUM_FRAMES=${NUM_FRAMES:-24}
MODE=${MODE:-auto}

CMD=(
  env
  PYTHONPATH="${PROJ}"
  "${PYTHON_BIN}"
  "${SCRIPT}"
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --num-frames "${NUM_FRAMES}"
  --mode "${MODE}"
  --force
)

if [[ -n "${MAX_FILES:-}" ]]; then
  CMD+=(--max-files "${MAX_FILES}")
fi
if [[ -n "${MAX_ROWS_PER_FILE:-}" ]]; then
  CMD+=(--max-rows-per-file "${MAX_ROWS_PER_FILE}")
fi
if [[ "${SKIP_SMOKE:-0}" == "1" ]]; then
  CMD+=(--skip-smoke)
fi

echo "[prepare] command: ${CMD[*]}"
exec "${CMD[@]}"
