#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 CONFIG GPU_ID WORKER_NAME" >&2
  exit 2
fi

CONFIG="$(realpath "$1")"
GPU_ID="$2"
WORKER_NAME="$3"
# shellcheck source=/dev/null
source "${CONFIG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="${RUN_ROOT}/generation/queue.tsv"
CURSOR="${RUN_ROOT}/generation/cursor"
LOCK="${RUN_ROOT}/generation/queue.lock"
STATE_DIR="${RUN_ROOT}/generation/task_state"
VALIDATION_DIR="${RUN_ROOT}/generation/validations"
LOG_DIR="${RUN_ROOT}/generation/logs"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
VERIFY="${SCRIPT_DIR}/verify_test5_ablation_outputs.py"
INPUT_LIST="${RUN_ROOT}/input_unique.txt"
WORKER_LOG="${LOG_DIR}/${WORKER_NAME}.log"

mkdir -p "${STATE_DIR}" "${VALIDATION_DIR}" "${LOG_DIR}"
exec > >(tee -a "${WORKER_LOG}") 2>&1

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

wait_for_gpu() {
  local used
  while true; do
    used="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [[ -n "${used}" && "${used}" -lt "${GEN_GPU_MAX_USED_MIB}" ]]; then
      return
    fi
    echo "[generation-worker] wait gpu=${GPU_ID} used_mib=${used:-unavailable}"
    sleep "${GPU_WAIT_SECONDS}"
  done
}

append_unique_result() {
  local destination="$1" task_id="$2"
  shift 2
  exec 8>"${RUN_ROOT}/generation/results.lock"
  flock 8
  if ! grep -q "^${task_id}"$'\t' "${destination}" 2>/dev/null; then
    {
      printf '%s' "${task_id}"
      printf '\t%s' "$@"
      printf '\n'
    } >> "${destination}"
  fi
  flock -u 8
  exec 8>&-
}

echo "[generation-worker] start worker=${WORKER_NAME} gpu=${GPU_ID}"
done_count=0
failed_count=0
while true; do
  task="$(claim_task)"
  [[ -n "${task}" ]] || break
  IFS=$'\t' read -r task_id model block head <<< "${task}"
  complete_state="${STATE_DIR}/${task_id}.complete"
  failed_state="${STATE_DIR}/${task_id}.failed"
  if [[ -f "${complete_state}" ]]; then
    echo "[generation-worker] skip completed ${task_id}"
    continue
  fi

  printf -v block_padded "%02d" "$((10#${block}))"
  printf -v head_padded "%02d" "$((10#${head}))"
  tag="self_attn_head_zero_block${block_padded}_head${head_padded}"
  task_log="${LOG_DIR}/${task_id}.log"
  validation="${VALIDATION_DIR}/${task_id}.json"

  common_env=(
    INPUT_LIST="${INPUT_LIST}"
    HEIGHT="${HEIGHT}" WIDTH="${WIDTH}" NUM_FRAMES="${NUM_FRAMES}"
    CONTEXT_FRAMES="${CONTEXT_FRAMES}"
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}"
    FPS="${FPS}" SEED="${SEED}" NEGATIVE_PROMPT="${NEGATIVE_PROMPT}"
  )
  case "${model}" in
    wan_lora|xssc)
      config_root="${OUTPUT_BASE}/${model}/${tag}"
      command=(
        env "${common_env[@]}"
        OUTPUT_BASE="${OUTPUT_BASE}"
        CFG_SCALE="${CFG_SCALE}"
        WAN_ROOT="${WAN_ROOT}"
        WAN_LORA_ROOT="${WAN_LORA_ROOT}"
        XSSC_WEIGHTS_ROOT="${XSSC_WEIGHTS_ROOT}"
        XSSC_ROOT="${XSSC_ROOT}"
        XSSC_CONFIG="${XSSC_CONFIG}"
        XSSC_CHECKPOINT="${XSSC_CHECKPOINT}"
        bash "${SCRIPT_DIR}/run_physiciq_one.sh"
        "${model}" self_attn_head_zero "${block}" "${GPU_ID}" "${head}"
      )
      ;;
    physrvg)
      config_root="${OUTPUT_BASE}/PhyRVG/${tag}"
      command=(
        env "${common_env[@]}"
        OUTPUT_BASE="${OUTPUT_BASE}/PhyRVG"
        GUIDANCE_SCALE="${GUIDANCE_SCALE}"
        DO_CFG="${PHYSRVG_DO_CFG}"
        PHYSRVG_ROOT="${PHYSRVG_ROOT}"
        MODEL_ID="${PHYSRVG_MODEL_ID}"
        DIT_CHECKPOINT="${PHYSRVG_DIT_CHECKPOINT}"
        LORA_CHECKPOINT="${PHYSRVG_LORA_CHECKPOINT}"
        bash "${SCRIPT_DIR}/run_physrvg_physiciq_one.sh"
        self_attn_head_zero "${block}" "${GPU_ID}" "${head}"
      )
      ;;
    *)
      echo "[generation-worker] invalid model=${model}"
      printf 'invalid model=%s\n' "${model}" > "${failed_state}"
      failed_count=$((failed_count + 1))
      continue
      ;;
  esac

  wait_for_gpu
  {
    echo "task_id=${task_id}"
    echo "model=${model}"
    echo "block=${block}"
    echo "head=${head}"
    echo "gpu=${GPU_ID}"
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "${task_log}"
  echo "[generation-worker] run ${task_id} model=${model} block=${block} head=${head}"

  status=0
  "${command[@]}" >> "${task_log}" 2>&1 || status=$?
  if [[ "${status}" -eq 0 ]]; then
    "${PYTHON}" "${VERIFY}" \
      --config-root "${config_root}" \
      --input-list "${INPUT_LIST}" \
      --model "${model}" \
      --mode self_attn_head_zero \
      --block "${block}" \
      --head "${head}" \
      --expected-cases "${EXPECTED_CASES}" \
      --output "${validation}" >> "${task_log}" 2>&1 || status=$?
  fi

  if [[ "${status}" -eq 0 ]]; then
    printf 'task_id=%s\nmodel=%s\nblock=%s\nhead=%s\ngpu=%s\n' \
      "${task_id}" "${model}" "${block}" "${head}" "${GPU_ID}" \
      > "${complete_state}"
    rm -f "${failed_state}"
    append_unique_result "${RUN_ROOT}/generation/completed.tsv" \
      "${task_id}" "${model}" "${block}" "${head}" "${WORKER_NAME}"
    done_count=$((done_count + 1))
  else
    printf 'task_id=%s\nmodel=%s\nblock=%s\nhead=%s\ngpu=%s\nstatus=%s\n' \
      "${task_id}" "${model}" "${block}" "${head}" "${GPU_ID}" "${status}" \
      > "${failed_state}"
    append_unique_result "${RUN_ROOT}/generation/failed.tsv" \
      "${task_id}" "${model}" "${block}" "${head}" "${WORKER_NAME}" "${status}"
    failed_count=$((failed_count + 1))
  fi
done

printf 'worker=%s\ngpu=%s\ndone=%s\nfailed=%s\nfinished_utc=%s\n' \
  "${WORKER_NAME}" "${GPU_ID}" "${done_count}" "${failed_count}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${RUN_ROOT}/generation/state/${WORKER_NAME}.worker.complete"
echo "[generation-worker] finish worker=${WORKER_NAME} done=${done_count} failed=${failed_count}"
