#!/usr/bin/env bash
set -euo pipefail

# Example:
#   CUDA_VISIBLE_DEVICES=4,5 \
#   /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots_physicIQ.sh

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

WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots/checkpoints/step-002000
INPUT_TXT=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_xSSC
METHOD_NAME=formal_mix49_b2_dropout_metrics_20260719T204359Z_step-002000_steps40_512x896_ctx08_49f_defaultnegprompt
RUN_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots.sh

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
  STEP_OUTPUT_DIR_NAME="${METHOD_NAME}" \
  SHARD_TAG="${shard_name}" \
  TRACE_ROOT="${trace_dir}" \
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
