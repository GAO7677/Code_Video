#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 CHECKPOINT_DIR [GPU_ID] [OUTPUT_ROOT]" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
GPU_ID="${2:-2}"
OUTPUT_ROOT="${3:-/data/gaoya/agent-data/outputs/xssc_dinov3_checkpoint_inference}"
TEST_LIST="${TEST_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
TRAIN_XSSC_DIR="${PROJ}/code_vjepa_vggt/train_xSSC"
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

XSSC_EXP_ROOT="${TRAIN_XSSC_DIR}/xssc_rsfq2_ytvis_dinov3_vitl16_256"
XSSC_ROOT="${XSSC_ROOT:-${XSSC_EXP_ROOT}}"
XSSC_CONFIG="${XSSC_CONFIG:-${XSSC_EXP_ROOT}/upstream/config-randsfq/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py}"
XSSC_CHECKPOINT="${XSSC_CHECKPOINT:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/restart_save1000_20260720T140029Z/movi_c_transfer15000_b64_acc3_20260721T134713Z/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/step-026000.pth}"
DINOV3_ROOT="${DINOV3_ROOT:-${XSSC_EXP_ROOT}/third_party/dinov3}"
DINOV3_CHECKPOINT="${DINOV3_CHECKPOINT:-/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
XSSC_BOX_CACHE_DIR="${XSSC_BOX_CACHE_DIR:-/data/gaoya/agent-data/cache/xssc_dinov3_context_amg_boxes_wan_infer}"
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
if [[ ! -s "${TEST_LIST}" ]]; then
  echo "TEST_LIST not found or empty: ${TEST_LIST}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${TRACE_ROOT}" "${XSSC_BOX_CACHE_DIR}"
exec env \
  PYTHONPATH="${PROJ}:${TRAIN_XSSC_DIR}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  XSSC_ROOT="${XSSC_ROOT}" \
  XSSC_CONFIG="${XSSC_CONFIG}" \
  XSSC_CHECKPOINT="${XSSC_CHECKPOINT}" \
  DINOV3_ROOT="${DINOV3_ROOT}" \
  DINOV3_CHECKPOINT="${DINOV3_CHECKPOINT}" \
  XSSC_BOX_SOURCE=amg \
  XSSC_BOX_CACHE_DIR="${XSSC_BOX_CACHE_DIR}" \
  XSSC_SLOT_TEMPORAL_MODE="${XSSC_SLOT_TEMPORAL_MODE:-full}" \
  "${PYTHON}" -m code_vjepa_vggt.train_xSSC.infer_xssc_context_slots_dinov3 \
  --weights-root "${CHECKPOINT_DIR}" \
  --input-json-list-path "${TEST_LIST}" \
  --model-name xssc_dinov3_ctx_slots_wan22_5b \
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
