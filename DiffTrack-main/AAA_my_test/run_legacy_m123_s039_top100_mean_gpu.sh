#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 GPU_ID WORKER_ID [NUM_WORKERS]" >&2
  exit 2
fi

GPU_ID=$1
WORKER_ID=$2
NUM_WORKERS=${3:-4}
if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU 4 is forbidden by workspace policy" >&2
  exit 2
fi

REPO=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER=${REPO}/AAA_my_test/run_legacy_m123_head_scope_s039_top100_mean.py
CAPTURE_ROOT=/data/gaoya/agent-data/outputs/object_query_attention_overlays/m123_head_scope_s039_top100_mean_v1
LOG_ROOT=${CAPTURE_ROOT}/logs

mkdir -p "${LOG_ROOT}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -u +%FT%TZ)] GPU=${GPU_ID} worker=${WORKER_ID}/${NUM_WORKERS} S039 fixed-F04 Top100 mean"
exec "${PYTHON}" -u "${WORKER}" \
  --worker-id "${WORKER_ID}" \
  --num-workers "${NUM_WORKERS}" \
  --capture-root "${CAPTURE_ROOT}" \
  --device cuda \
  2>&1 | tee -a "${LOG_ROOT}/gpu${GPU_ID}_worker${WORKER_ID}.log"
