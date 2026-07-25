#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 GPU_ID WORKER_NAME RUN_ROOT OUTPUT_BASE INPUT_LIST" >&2
  exit 2
fi

GPU_ID="$1"
WORKER_NAME="$2"
RUN_ROOT="$3"
OUTPUT_BASE="$4"
INPUT_LIST="$5"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/remaining_blocks_experiment.env"
MANAGER="${SCRIPT_DIR}/manage_remaining_block_pipeline.py"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
QUEUE="${RUN_ROOT}/generation/queue.tsv"
CURSOR="${RUN_ROOT}/generation/cursor"
LOCK="${RUN_ROOT}/generation/queue.lock"
LOG="${RUN_ROOT}/generation/logs/${WORKER_NAME}.log"

if [[ ! -s "${CONFIG}" ]]; then
  echo "Missing experiment config: ${CONFIG}" >&2
  exit 2
fi
set -a
source "${CONFIG}"
set +a

mkdir -p "$(dirname -- "${LOG}")" "${RUN_ROOT}/generation/state" \
  "${RUN_ROOT}/generation/validations"
exec > >(tee -a "${LOG}") 2>&1

claim_task() {
  local line_number task
  exec 9>"${LOCK}"
  flock 9
  line_number="$(<"${CURSOR}")"
  task="$(sed -n "${line_number}p" "${QUEUE}")"
  if [[ -n "${task}" ]]; then
    printf '%s\n' "$((line_number + 1))" > "${CURSOR}"
  fi
  flock -u 9
  exec 9>&-
  printf '%s' "${task}"
}

num_done=0
num_failed=0
echo "[generation-worker] start worker=${WORKER_NAME} gpu=${GPU_ID}"
while true; do
  task="$(claim_task)"
  [[ -z "${task}" ]] && break
  IFS=$'\t' read -r task_id model mode block config_root <<< "${task}"
  echo "[generation-worker] task=${task_id} model=${model} mode=${mode} block=${block}"

  set +e
  if [[ "${model}" == "physrvg" ]]; then
    PYTHON="${PHYSRVG_PYTHON}" \
    INPUT_LIST="${INPUT_LIST}" \
    OUTPUT_BASE="${OUTPUT_BASE}/PhyRVG" \
    PHYSRVG_ROOT="${PHYSRVG_ROOT}" \
    MODEL_ID="${PHYSRVG_MODEL_ID}" \
    DIT_CHECKPOINT="${PHYSRVG_DIT_CHECKPOINT}" \
    LORA_CHECKPOINT="${PHYSRVG_LORA_CHECKPOINT}" \
    LIMIT="" \
      bash "${SCRIPT_DIR}/run_physrvg_physiciq_one.sh" \
      "${mode}" "${block}" "${GPU_ID}"
  else
    PYTHON="${WAN_PYTHON}" \
    INPUT_LIST="${INPUT_LIST}" \
    OUTPUT_BASE="${OUTPUT_BASE}" \
    WAN_ROOT="${WAN_ROOT}" \
    WAN_LORA_ROOT="${WAN_LORA_ROOT}" \
    XSSC_WEIGHTS_ROOT="${XSSC_WEIGHTS_ROOT}" \
    XSSC_ROOT="${XSSC_ROOT}" \
    XSSC_CONFIG="${XSSC_CONFIG}" \
    XSSC_CHECKPOINT="${XSSC_CHECKPOINT}" \
    HEIGHT="${HEIGHT}" \
    WIDTH="${WIDTH}" \
    NUM_FRAMES="${NUM_FRAMES}" \
    CONTEXT_FRAMES="${CONTEXT_FRAMES}" \
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
    CFG_SCALE="${CFG_SCALE}" \
    FPS="${FPS}" \
    SEED="${SEED}" \
    LIMIT="" \
    NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
      bash "${SCRIPT_DIR}/run_physiciq_one.sh" \
      "${model}" "${mode}" "${block}" "${GPU_ID}"
  fi
  status=$?
  if [[ "${status}" -eq 0 ]]; then
    "${PYTHON_BIN}" "${MANAGER}" validate-config \
      --config-root "${config_root}" \
      --input-list "${INPUT_LIST}" \
      --output "${RUN_ROOT}/generation/validations/${task_id}.json"
    status=$?
  fi
  set -e

  if [[ "${status}" -eq 0 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${model}" "${mode}" "${block}" "${config_root}" "${WORKER_NAME}" \
      >> "${RUN_ROOT}/generation/completed.tsv"
    num_done=$((num_done + 1))
  else
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${model}" "${mode}" "${block}" "${config_root}" "${WORKER_NAME}" "${status}" \
      >> "${RUN_ROOT}/generation/failed.tsv"
    num_failed=$((num_failed + 1))
  fi
done

printf 'worker=%s\ngpu=%s\ndone=%s\nfailed=%s\nfinished_utc=%s\n' \
  "${WORKER_NAME}" "${GPU_ID}" "${num_done}" "${num_failed}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${RUN_ROOT}/generation/state/${WORKER_NAME}.complete"
echo "[generation-worker] finish worker=${WORKER_NAME} done=${num_done} failed=${num_failed}"
