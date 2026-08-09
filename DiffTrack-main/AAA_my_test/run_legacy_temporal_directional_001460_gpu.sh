#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU WORKER_ID NUM_WORKERS}"
WORKER_ID="${2:?usage: $0 GPU WORKER_ID NUM_WORKERS}"
NUM_WORKERS="${3:?usage: $0 GPU WORKER_ID NUM_WORKERS}"
[[ "${GPU}" =~ ^[0-3]$ ]] || { echo "GPU must be one of 0,1,2,3 (GPU4 is forbidden)" >&2; exit 2; }
[[ "${WORKER_ID}" =~ ^[0-9]+$ ]] || { echo "WORKER_ID must be a non-negative integer" >&2; exit 2; }
[[ "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_WORKERS must be positive" >&2; exit 2; }

ROOT="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
OUTPUT_BASE="/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326"
OUTPUT_ROOT="${OUTPUT_BASE}/attention_matrix_ablations_temporal_tube_v1"
LOG_ROOT="${OUTPUT_BASE}/temporal_directional_logs"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/gpu${GPU}_worker${WORKER_ID}.log") 2>&1
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${ROOT}"

echo "[$(date -u +%FT%TZ)] GPU${GPU} worker ${WORKER_ID}/${NUM_WORKERS} start"
"${PYTHON}" -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py \
  --case 0613pybullet_sample_001460_w002 \
  --seed 47326 \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  --worker-id "${WORKER_ID}" \
  --num-workers "${NUM_WORKERS}" \
  --mask-modes \
    self_future incoming_future outgoing_future \
    self_past incoming_past outgoing_past
echo "[$(date -u +%FT%TZ)] GPU${GPU} worker ${WORKER_ID}/${NUM_WORKERS} complete"
