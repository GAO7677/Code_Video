#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 GPU_ID WORKER_ID NUM_WORKERS [TASK_INDEX]" >&2
  exit 2
fi

GPU_ID=$1
WORKER_ID=$2
NUM_WORKERS=$3
TASK_INDEX=${4:-}
if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU 4 is forbidden by workspace policy" >&2
  exit 2
fi

REPO=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER=${REPO}/AAA_my_test/run_legacy_m123_s039_query_receiver.py
CAPTURE_ROOT=/data/gaoya/agent-data/outputs/object_query_attention_overlays/m123_head_scope_s039_query_receiver_v1
LOG_ROOT=${CAPTURE_ROOT}/logs

mkdir -p "${LOG_ROOT}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARGS=(
  --worker-id "${WORKER_ID}"
  --num-workers "${NUM_WORKERS}"
  --capture-root "${CAPTURE_ROOT}"
  --device cuda
)
LABEL=worker${WORKER_ID}
if [[ -n "${TASK_INDEX}" ]]; then
  ARGS+=(--task-index "${TASK_INDEX}")
  LABEL=pilot_task${TASK_INDEX}
fi

echo "[$(date -u +%FT%TZ)] GPU=${GPU_ID} ${LABEL} S039 query receiver S(q)/E(q)"
exec "${PYTHON}" -u "${WORKER}" "${ARGS[@]}" \
  2>&1 | tee -a "${LOG_ROOT}/gpu${GPU_ID}_${LABEL}.log"
