#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${ROOT}/AAA_my_test/run_top5_head_zero_ablation_worker.py"
OUTPUT="/data/gaoya/agent-data/outputs/top5_pck_head_zero_ablation_5case"
mkdir -p "${OUTPUT}/logs"

CUDA_VISIBLE_DEVICES=2 "${PYTHON}" "${WORKER}" --model baseline --device cuda:0 \
  >"${OUTPUT}/logs/baseline_gpu2.log" 2>&1 &
PID_BASE=$!
CUDA_VISIBLE_DEVICES=5 "${PYTHON}" "${WORKER}" --model lora --device cuda:0 \
  >"${OUTPUT}/logs/lora_gpu5.log" 2>&1 &
PID_LORA=$!
trap 'kill "${PID_BASE}" "${PID_LORA}" 2>/dev/null || true' INT TERM EXIT

STATUS=0
wait "${PID_BASE}" || STATUS=$?
wait "${PID_LORA}" || STATUS=$?
trap - INT TERM EXIT
exit "${STATUS}"
