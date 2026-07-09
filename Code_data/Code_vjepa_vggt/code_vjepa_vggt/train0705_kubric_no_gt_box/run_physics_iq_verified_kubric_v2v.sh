#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main}"
RUNNER_PY="${SCRIPT_DIR}/run_physics_iq_verified_kubric_v2v.py"

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000}"
MODEL_NAME="${MODEL_NAME:-train_stage1b_kubric0708_physiq_verified}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/physics_iq_verified_v2v}"
VERIFIED_ROOT="${VERIFIED_ROOT:-/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified}"
DESCRIPTIONS_FILE="${DESCRIPTIONS_FILE:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv}"

FPS="${FPS:-30}"
NUM_FRAMES="${NUM_FRAMES:-150}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-20}"
SAMPLING_MODE="${SAMPLING_MODE:-prefix}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
CFG_SCALE="${CFG_SCALE:-5.0}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-}"
LIMIT="${LIMIT:-}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0,1}}"
INFERENCE_DEVICES="${INFERENCE_DEVICES:-cuda:0,cuda:1}"

export PYTHONPATH="${REPO_ROOT}:${DIFFSYNTH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

CMD=(
  "${PYTHON_BIN}"
  "${RUNNER_PY}"
  --weights-root "${WEIGHTS_ROOT}"
  --model-name "${MODEL_NAME}"
  --output-root "${OUTPUT_ROOT}"
  --verified-root "${VERIFIED_ROOT}"
  --descriptions-file "${DESCRIPTIONS_FILE}"
  --fps "${FPS}"
  --num-frames "${NUM_FRAMES}"
  --context-frames "${CONTEXT_FRAMES}"
  --sampling-mode "${SAMPLING_MODE}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --cfg-scale "${CFG_SCALE}"
  --seed "${SEED}"
  --inference-devices "${INFERENCE_DEVICES}"
)

if [[ -n "${RUN_NAME}" ]]; then
  CMD+=(--run-name "${RUN_NAME}")
fi

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

CMD+=("$@")

echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] ${CMD[*]}"
"${CMD[@]}"
