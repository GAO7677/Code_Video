#!/usr/bin/env bash
set -euo pipefail

# Example:
# CUDA_VISIBLE_DEVICES=5,6 \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_randomcrop_pooled_physicIQ.sh

cd /home/gaoya

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set, for example: CUDA_VISIBLE_DEVICES=4,5 $0" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_SHARDS="${#GPU_IDS[@]}"
if (( NUM_SHARDS <= 0 )); then
  echo "No GPU ids parsed from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi



RUN_NAME="xssc_randomcrop_pooled_gpu45_mix49_formal_randomcrop_pooled_gpu45_20260720T110031Z"
STEP_NAME="${STEP_NAME:-step-002000}"
WEIGHTS_ROOT="/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/${RUN_NAME}/checkpoints/${STEP_NAME}"

NEGATIVE_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

METHOD_NAME="xssc_randomcrop_pooled_${STEP_NAME}_steps40_512x896_ctx08_49f_defaultnegprompt"
# METHOD_NAME="${RUN_NAME}_${STEP_NAME}_steps40_512x896_ctx08_49f_nullnegprompt"




INPUT_TXT=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_xSSC

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

META_ROOT="${OUTPUT_ROOT}/_run_meta/${METHOD_NAME}"
mkdir -p "${META_ROOT}/shards" "${META_ROOT}/logs" "${META_ROOT}/numeric_traces"

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_name="$(printf 'shard_%02d' "${shard_idx}")"
  awk -v idx="${shard_idx}" -v n="${NUM_SHARDS}" 'NF && ((NR - 1) % n == idx)' \
    "${INPUT_TXT}" > "${META_ROOT}/shards/${shard_name}.txt"
done

pids=()
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu_id="${GPU_IDS[$shard_idx]}"
  gpu_id="${gpu_id//[[:space:]]/}"
  if [[ -z "${gpu_id}" ]]; then
    echo "Empty GPU id at shard index ${shard_idx}" >&2
    exit 2
  fi

  shard_name="$(printf 'shard_%02d' "${shard_idx}")"
  shard_file="${META_ROOT}/shards/${shard_name}.txt"
  log_file="${META_ROOT}/logs/${shard_name}.log"
  trace_dir="${META_ROOT}/numeric_traces/${shard_name}"

  TEST_LIST="${shard_file}" \
  NUM_INFERENCE_STEPS=40 \
  STEP_OUTPUT_DIR_NAME="${METHOD_NAME}" \
  SHARD_TAG="${shard_name}" \
  TRACE_ROOT="${trace_dir}" \
  NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
  XSSC_TRAIN_CROP_MODE=random \
  XSSC_EVAL_CROP_MODE=center \
  XSSC_SLOT_PERTURB=none \
  bash "${RUN_SCRIPT}" \
    "${WEIGHTS_ROOT}" \
    "${gpu_id}" \
    "${OUTPUT_ROOT}" \
    > "${log_file}" 2>&1 &

  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
