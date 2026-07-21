#!/usr/bin/env bash
set -euo pipefail

# Reproduce the test_5 inference layout produced by
# watch_xssc_randomcrop_pooled_checkpoints.sh for the random-crop pooled run.

cd /home/gaoya

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set, for example: CUDA_VISIBLE_DEVICES=6 $0" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
GPU_ID="${GPU_IDS[0]//[[:space:]]/}"
if [[ -z "${GPU_ID}" ]]; then
  echo "No GPU id parsed from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

RUN_NAME=xssc_randomcrop_pooled_gpu45_mix49_formal_randomcrop_pooled_gpu45_20260720T110031Z
STEP_NAME="${STEP_NAME:-step-002000}"
RUN_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/${RUN_NAME}
WEIGHTS_ROOT="${RUN_ROOT}/checkpoints/${STEP_NAME}"
INPUT_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt
OUTPUT_BASE=/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/test_5/${RUN_NAME}
OUTPUT_ROOT="${OUTPUT_BASE}/${STEP_NAME}"
RUN_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_randomcrop_pooled_slots.sh

if [[ ! -d "${WEIGHTS_ROOT}" ]]; then
  echo "WEIGHTS_ROOT not found: ${WEIGHTS_ROOT}" >&2
  exit 2
fi
if [[ ! -s "${INPUT_TXT}" ]]; then
  echo "INPUT_TXT not found or empty: ${INPUT_TXT}" >&2
  exit 2
fi
if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "RUN_SCRIPT not found: ${RUN_SCRIPT}" >&2
  exit 2
fi

TEST_LIST="${INPUT_TXT}" \
NUM_INFERENCE_STEPS=40 \
STEP_OUTPUT_DIR_NAME="${STEP_NAME}" \
XSSC_TRAIN_CROP_MODE=random \
XSSC_EVAL_CROP_MODE=center \
XSSC_SLOT_PERTURB=none \
bash "${RUN_SCRIPT}" \
  "${WEIGHTS_ROOT}" \
  "${GPU_ID}" \
  "${OUTPUT_ROOT}"
