#!/usr/bin/env bash
set -euo pipefail

GPU_ID=3
REPO=/home/gaoya/Code_Video/DiffTrack-main
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
RUNNER="${REPO}/AAA_my_test/object_query_ablation_metrics/training_free_m1_control/run_multi_object_guidance_search.py"
SOURCE_ROOT=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_m1_multi_object_search_v1
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_m2_multi_object_search_v1
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
echo "[$(date -u +%FT%TZ)] start GPU=${GPU_ID} M2 001460 5seeds_x_16" | tee -a "${LOG}"
"${PYTHON_BIN}" -u "${RUNNER}" \
  --worker-id 0 \
  --num-workers 1 \
  --stage guidance \
  --manifest-path "${SOURCE_ROOT}/search_manifest.json" \
  --output-root "${OUTPUT_ROOT}" \
  --tracks-root "${SOURCE_ROOT}/tracks" \
  --device cuda \
  --perturbation-mode m2_multi_object_independent \
  --case 0613pybullet_sample_001460_w002 \
  2>&1 | tee -a "${LOG}"
echo "[$(date -u +%FT%TZ)] complete GPU=${GPU_ID} M2 001460" | tee -a "${LOG}"
