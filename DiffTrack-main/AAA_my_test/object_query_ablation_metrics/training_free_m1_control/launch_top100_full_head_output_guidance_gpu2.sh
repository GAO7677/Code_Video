#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
DRY_RUN=()
if (( $# == 1 )); then
  if [[ "$1" != "--dry-run" ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
  fi
  DRY_RUN=(--dry-run)
fi

GPU_ID=2
REPO=/home/gaoya/Code_Video/DiffTrack-main
CONTROL_DIR="${REPO}/AAA_my_test/object_query_ablation_metrics/training_free_m1_control"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
RUNNER="${CONTROL_DIR}/run_multi_object_guidance_search.py"
SOURCE_ROOT=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_m1_multi_object_search_v1
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_top100_full_head_output_zero_search_v1
MANIFEST="${SOURCE_ROOT}/search_manifest.json"
TRACKS_ROOT="${SOURCE_ROOT}/tracks"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
cd "${REPO}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/data/gaoya/agent-data/cache/huggingface
export TORCH_HOME=/data/gaoya/agent-data/cache/torch
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

LOG="${LOG_ROOT}/gpu${GPU_ID}_worker0.log"
echo "[$(date -u +%FT%TZ)] start GPU=${GPU_ID} strict_0613_5cases_x_5seeds" | tee -a "${LOG}"
"${PYTHON_BIN}" -u "${RUNNER}" \
  --worker-id 0 \
  --num-workers 1 \
  --stage guidance \
  --manifest-path "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --tracks-root "${TRACKS_ROOT}" \
  --device cuda \
  --perturbation-mode full_head_output_zero \
  --case 0613pybullet_sample_000301_w000 \
  --case 0613pybullet_sample_000331_w001 \
  --case 0613pybullet_sample_000336_w001 \
  --case 0613pybullet_sample_001455_w000 \
  --case 0613pybullet_sample_001460_w002 \
  "${DRY_RUN[@]}" 2>&1 | tee -a "${LOG}"
echo "[$(date -u +%FT%TZ)] complete GPU=${GPU_ID} strict_0613_5cases_x_5seeds" | tee -a "${LOG}"
