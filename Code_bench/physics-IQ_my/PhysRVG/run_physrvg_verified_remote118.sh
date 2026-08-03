#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/physrvg_verified_remote118.env}"
# shellcheck source=/dev/null
source "${CONFIG}"

ACTION="${1:-all}"
MAX_ITEMS="${MAX_ITEMS:-}"
FORCE="${FORCE:-0}"
GPU="${GPU:-}"

case "${ACTION}" in
  generate|score|all) ;;
  *) echo "usage: GPU=<id> $0 {generate|score|all}" >&2; exit 2 ;;
esac

if [[ "${ACTION}" != "score" ]]; then
  if [[ -z "${GPU}" ]]; then
    echo "[error] set GPU explicitly, for example GPU=3" >&2
    exit 2
  fi
  if [[ "${GPU}" == "4" ]]; then
    echo "[error] GPU 4 is prohibited in this workspace" >&2
    exit 2
  fi
fi

echo "[sync] adapter -> ${REMOTE_HOST}:${REMOTE_ADAPTER_DIR}"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_ADAPTER_DIR}' '${REMOTE_INPUT_ROOT}' '${REMOTE_OUTPUT_ROOT}'"
rsync -a "${LOCAL_ADAPTER_DIR}/" "${REMOTE_HOST}:${REMOTE_ADAPTER_DIR}/"

if [[ "${ACTION}" != "score" ]]; then
  echo "[sync] exact xSSC BPP/V2V inputs -> ${REMOTE_HOST}:${REMOTE_INPUT_ROOT}"
  rsync -a "${LOCAL_INPUT_ROOT}/" "${REMOTE_HOST}:${REMOTE_INPUT_ROOT}/"

  generate_cmd=(
    "${REMOTE_PHYSRVG_PYTHON}" "${REMOTE_ADAPTER_DIR}/generate_physrvg_verified.py"
    --physrvg-root "${REMOTE_PHYSRVG_ROOT}"
    --model-id "${REMOTE_MODEL_ID}"
    --dit-checkpoint "${REMOTE_DIT_CHECKPOINT}"
    --lora-checkpoint "${REMOTE_LORA_CHECKPOINT}"
    --input-list "${REMOTE_INPUT_LIST}"
    --output-root "${REMOTE_OUTPUT_ROOT}"
    --run-name "${RUN_NAME}"
    --device cuda:0
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --condition-fps "${CONDITION_FPS}"
    --condition-frames "${CONDITION_FRAMES}"
    --model-context-frames "${MODEL_CONTEXT_FRAMES}"
    --model-chunk-frames "${MODEL_CHUNK_FRAMES}"
    --clean-prefix-frames "${CLEAN_PREFIX_FRAMES}"
    --model-fps "${MODEL_FPS}"
    --target-fps "${TARGET_FPS}"
    --target-frames "${TARGET_FRAMES}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --seed "${SEED}"
  )
  [[ -n "${MAX_ITEMS}" ]] && generate_cmd+=(--max-items "${MAX_ITEMS}")
  [[ "${FORCE}" == "1" ]] && generate_cmd+=(--force)
  printf -v remote_generate '%q ' "${generate_cmd[@]}"
  echo "[generate] host=${REMOTE_HOST} physical_gpu=${GPU} run=${RUN_NAME}"
  ssh -t "${REMOTE_HOST}" "cd '${REMOTE_PHYSRVG_ROOT}' && CUDA_VISIBLE_DEVICES='${GPU}' PYTHONNOUSERSITE=1 ${remote_generate}"
fi

if [[ "${ACTION}" != "generate" ]]; then
  run_dir="${REMOTE_OUTPUT_ROOT}/${RUN_NAME}"
  result_csv="${REMOTE_EVAL_ROOT}/physics-IQ-benchmark-verified/results/${RUN_NAME}.csv"
  printf -v remote_score '%q ' \
    "${REMOTE_EVAL_PYTHON}" "${REMOTE_PHYSIQ_ROOT}/physiq/run_physics_iq.py" \
    --input_folders "${run_dir}" \
    --output_folder "${REMOTE_EVAL_ROOT}" \
    --descriptions_file "${REMOTE_DESCRIPTIONS_FILE}" \
    --benchmark_base_folder "${REMOTE_BENCHMARK_BASE}"
  printf -v remote_aggregate '%q ' \
    "${REMOTE_EVAL_PYTHON}" "${REMOTE_PHYSIQ_ROOT}/physiq/aggregate_runs_from_csvs.py" \
    "${result_csv}" --score-type verified
  echo "[score] official Physics-IQ Verified evaluator on ${REMOTE_HOST}"
  ssh -t "${REMOTE_HOST}" "cd '${REMOTE_PHYSIQ_ROOT}' && PYTHONNOUSERSITE=1 ${remote_score} && ${remote_aggregate}"
fi
