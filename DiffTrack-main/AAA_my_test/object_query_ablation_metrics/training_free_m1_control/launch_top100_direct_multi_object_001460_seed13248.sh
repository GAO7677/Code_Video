#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 GPU_ID [--dry-run]" >&2
  exit 2
fi

GPU_ID="$1"
DRY_RUN=()
if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU 4 is excluded by workspace policy." >&2
  exit 2
fi
if (( $# == 2 )); then
  if [[ "$2" != "--dry-run" ]]; then
    echo "Usage: $0 GPU_ID [--dry-run]" >&2
    exit 2
  fi
  DRY_RUN=(--dry-run)
fi

REPO=/home/gaoya/Code_Video/DiffTrack-main
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
RUNNER="${REPO}/AAA_my_test/object_query_ablation_metrics/training_free_m1_control/run_multi_object_guidance_search.py"
SOURCE_ROOT=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_m1_multi_object_search_v1
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_direct_multi_object_ablation_v1
LOG_ROOT="${OUTPUT_ROOT}/logs"
CASE=0613pybullet_sample_001460_w002
SEED=13248

mkdir -p "${LOG_ROOT}"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/data/gaoya/agent-data/cache/huggingface
export TORCH_HOME=/data/gaoya/agent-data/cache/torch
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

MODES=(
  m1_multi_object_blockdiag
  m2_multi_object_independent
  m3_multi_object_independent
  full_head_output_zero
)

for MODE in "${MODES[@]}"; do
  LOG="${LOG_ROOT}/gpu${GPU_ID}_${MODE}.log"
  echo "[$(date -u +%FT%TZ)] start GPU=${GPU_ID} direct ${MODE} ${CASE} seed=${SEED}" | tee -a "${LOG}"
  "${PYTHON_BIN}" -u "${RUNNER}" \
    --worker-id 0 \
    --num-workers 1 \
    --stage guidance \
    --execution-mode direct \
    --perturbation-mode "${MODE}" \
    --manifest-path "${SOURCE_ROOT}/search_manifest.json" \
    --output-root "${OUTPUT_ROOT}" \
    --tracks-root "${SOURCE_ROOT}/tracks" \
    --device cuda \
    --case "${CASE}" \
    --seed "${SEED}" \
    "${DRY_RUN[@]}" 2>&1 | tee -a "${LOG}"
  echo "[$(date -u +%FT%TZ)] complete GPU=${GPU_ID} direct ${MODE}" | tee -a "${LOG}"
done
