#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 GPU_ID WORKER_ID" >&2
  exit 2
fi

GPU_ID="$1"
WORKER_ID="$2"
if [[ "${GPU_ID}" != "1" && "${GPU_ID}" != "2" ]]; then
  echo "This frozen launch is restricted to physical GPU 1 or 2." >&2
  exit 2
fi
if [[ "${WORKER_ID}" != "0" && "${WORKER_ID}" != "1" ]]; then
  echo "WORKER_ID must be 0 or 1." >&2
  exit 2
fi

REPO=/home/gaoya/Code_Video/DiffTrack-main
CONTROL_DIR="${REPO}/AAA_my_test/object_query_ablation_metrics/training_free_m1_control"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BUILDER="${CONTROL_DIR}/build_multi_object_guidance_search_manifest.py"
RUNNER="${CONTROL_DIR}/run_multi_object_guidance_search.py"
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_m1_multi_object_search_v1
MANIFEST="${OUTPUT_ROOT}/search_manifest.json"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
cd "${REPO}"
if [[ ! -f "${MANIFEST}" ]]; then
  "${PYTHON_BIN}" "${BUILDER}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/data/gaoya/agent-data/cache/huggingface
export TORCH_HOME=/data/gaoya/agent-data/cache/torch
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

LOG="${LOG_ROOT}/gpu${GPU_ID}_worker${WORKER_ID}.log"
echo "[$(date -u +%FT%TZ)] start GPU=${GPU_ID} worker=${WORKER_ID}/2" | tee -a "${LOG}"
"${PYTHON_BIN}" -u "${RUNNER}" \
  --worker-id "${WORKER_ID}" \
  --num-workers 2 \
  --stage all \
  --manifest-path "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --tracks-root "${OUTPUT_ROOT}/tracks" \
  --device cuda \
  --group-batch-size 4 2>&1 | tee -a "${LOG}"
echo "[$(date -u +%FT%TZ)] complete GPU=${GPU_ID} worker=${WORKER_ID}/2" | tee -a "${LOG}"
