#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots_pybullet0717_test5.sh \
#   /path/to/checkpoints/step-000500 4

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 CHECKPOINT_DIR [GPU_ID] [OUTPUT_ROOT]" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
GPU_ID="${2:-4}"
OUTPUT_ROOT="${3:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/test_5/pybullet0717_xssc_context_slots}"
TEST_LIST="${TEST_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
XSSC_ROOT="${XSSC_ROOT:-/home/gaoya/Code_Video/xSSC-main}"
XSSC_CONFIG="${XSSC_CONFIG:-${XSSC_ROOT}/config-randsfq/rsfq2_r-ytvis.py}"
XSSC_CHECKPOINT="${XSSC_CHECKPOINT:-/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth}"
TRACE_ROOT="${TRACE_ROOT:-${OUTPUT_ROOT}/numeric_traces/$(basename "${CHECKPOINT_DIR}")}"

EXTRA_ARGS=()
if [ -n "${STEP_OUTPUT_DIR_NAME:-}" ]; then
  EXTRA_ARGS+=(--step-output-dir-name "${STEP_OUTPUT_DIR_NAME}")
fi
if [ -n "${SHARD_TAG:-}" ]; then
  EXTRA_ARGS+=(--shard-tag "${SHARD_TAG}")
fi
if [ "${NEGATIVE_PROMPT+x}" = x ]; then
  EXTRA_ARGS+=(--negative-prompt "${NEGATIVE_PROMPT}")
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "CHECKPOINT_DIR not found: ${CHECKPOINT_DIR}" >&2
  exit 2
fi
if [[ ! -s "${CHECKPOINT_DIR}/checkpoint.safetensors" ]]; then
  echo "checkpoint.safetensors not found under CHECKPOINT_DIR: ${CHECKPOINT_DIR}" >&2
  exit 2
fi
if [[ ! -s "${TEST_LIST}" ]]; then
  echo "TEST_LIST not found or empty: ${TEST_LIST}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${TRACE_ROOT}"
exec env \
  PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  XSSC_ROOT="${XSSC_ROOT}" \
  XSSC_CONFIG="${XSSC_CONFIG}" \
  XSSC_CHECKPOINT="${XSSC_CHECKPOINT}" \
  "${PYTHON}" -m code_vjepa_vggt.train_xSSC.infer_xssc_context_slots \
  --weights-root "${CHECKPOINT_DIR}" \
  --input-json-list-path "${TEST_LIST}" \
  --model-name xssc_pybullet0717_ctx_slots_wan22_5b \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --aux-device cuda:0 \
  --inference-devices cuda:0,cuda:0 \
  --height 512 \
  --width 896 \
  --num-frames 49 \
  --context-frames 8 \
  --sampling-mode prefix \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --dump-numeric-trace-root "${TRACE_ROOT}" \
  --force \
  "${EXTRA_ARGS[@]}"
