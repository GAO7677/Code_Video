#!/usr/bin/env bash
set -euo pipefail

# Example:
# CUDA_VISIBLE_DEVICES=2,3 \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots_dinov3_physicIQ.sh
#
# Override the default/latest checkpoint:
# WEIGHTS_ROOT=/path/to/checkpoints/step-001000 CUDA_VISIBLE_DEVICES=2 bash "$0"

cd /home/gaoya

TRAIN_XSSC_DIR=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC
RUN_SCRIPT="${RUN_SCRIPT:-${TRAIN_XSSC_DIR}/run_infer_xssc_context_slots_dinov3.sh}"

CHECKPOINT_BASE="${CHECKPOINT_BASE:-/data/gaoya/agent-data/checkpoints/train_xssc_context_slots_dinov3/formal_gpu01_20260722T143309Z/checkpoints}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-}"
INPUT_TXT="${INPUT_TXT:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_xSSC}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
XSSC_SLOT_TEMPORAL_MODE="${XSSC_SLOT_TEMPORAL_MODE:-full}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理}"

if [[ -z "${WEIGHTS_ROOT}" ]]; then
  if [[ ! -d "${CHECKPOINT_BASE}" ]]; then
    echo "CHECKPOINT_BASE not found: ${CHECKPOINT_BASE}" >&2
    exit 2
  fi
  WEIGHTS_ROOT="$(find "${CHECKPOINT_BASE}" -maxdepth 1 -type d -name 'step-*' | sort -V | tail -n 1)"
fi

if [[ -z "${WEIGHTS_ROOT}" ]]; then
  echo "No checkpoint found under CHECKPOINT_BASE=${CHECKPOINT_BASE}" >&2
  exit 2
fi

CHECKPOINT_NAME="$(basename "${WEIGHTS_ROOT}")"
METHOD_NAME="${METHOD_NAME:-dinov3_xssc_wan_${CHECKPOINT_NAME}_steps${NUM_INFERENCE_STEPS}_512x896_ctx08_49f_physicIQ_customprompt}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set, for example: CUDA_VISIBLE_DEVICES=2,3 $0" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_SHARDS="${#GPU_IDS[@]}"
if (( NUM_SHARDS <= 0 )); then
  echo "No GPU ids parsed from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

if [[ ! -d "${WEIGHTS_ROOT}" ]]; then
  echo "WEIGHTS_ROOT not found: ${WEIGHTS_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${WEIGHTS_ROOT}/checkpoint.safetensors" ]]; then
  echo "checkpoint.safetensors not found under WEIGHTS_ROOT: ${WEIGHTS_ROOT}" >&2
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

echo "DINOv3 Wan PhysicIQ inference"
echo "  WEIGHTS_ROOT=${WEIGHTS_ROOT}"
echo "  INPUT_TXT=${INPUT_TXT}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  METHOD_NAME=${METHOD_NAME}"
echo "  NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  NUM_SHARDS=${NUM_SHARDS}"

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
  shard_cases="$(wc -l < "${shard_file}")"

  echo "Launching ${shard_name}: gpu=${gpu_id}, cases=${shard_cases}, log=${log_file}"

  TEST_LIST="${shard_file}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  STEP_OUTPUT_DIR_NAME="${METHOD_NAME}" \
  SHARD_TAG="${shard_name}" \
  TRACE_ROOT="${trace_dir}" \
  NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
  XSSC_SLOT_TEMPORAL_MODE="${XSSC_SLOT_TEMPORAL_MODE}" \
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

if (( status == 0 )); then
  echo "Done. Videos are under: ${OUTPUT_ROOT}/${METHOD_NAME}"
else
  echo "One or more shards failed. Check logs under: ${META_ROOT}/logs" >&2
fi

exit "${status}"
