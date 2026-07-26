#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 CONFIG GPU_ID WORKER_NAME 'MODEL ...'" >&2
  exit 2
fi

CONFIG="$(realpath "$1")"
GPU_ID="$2"
WORKER_NAME="$3"
MODEL_TEXT="$4"
# shellcheck source=/dev/null
source "${CONFIG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
VERIFY="${SCRIPT_DIR}/verify_test5_ablation_outputs.py"
LOG_DIR="${BASELINE_RUN_ROOT}/logs"
STATE_DIR="${BASELINE_RUN_ROOT}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}" "${BASELINE_RUN_ROOT}/validations"
exec > >(tee -a "${LOG_DIR}/${WORKER_NAME}.log") 2>&1

echo "[baseline-worker] waiting for ${EXPECTED_HEAD_CONFIGS} head configurations"
while true; do
  completed="$(find "${HEAD_RUN_ROOT}/generation/task_state" -maxdepth 1 \
    -type f -name '*.complete' | wc -l)"
  failed="$(find "${HEAD_RUN_ROOT}/generation/task_state" -maxdepth 1 \
    -type f -name '*.failed' | wc -l)"
  if [[ "${failed}" -gt 0 ]]; then
    echo "[baseline-worker] head generation has ${failed} failed configurations"
    exit 1
  fi
  if [[ "${completed}" -ge "${EXPECTED_HEAD_CONFIGS}" ]]; then
    break
  fi
  echo "[baseline-worker] head generation=${completed}/${EXPECTED_HEAD_CONFIGS}"
  sleep "${WAIT_SECONDS}"
done

common_env=(
  HEIGHT="${HEIGHT}" WIDTH="${WIDTH}" NUM_FRAMES="${NUM_FRAMES}"
  CONTEXT_FRAMES="${CONTEXT_FRAMES}"
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}"
  FPS="${FPS}" SEED="${SEED}" NEGATIVE_PROMPT="${NEGATIVE_PROMPT}"
)

for model in ${MODEL_TEXT}; do
  complete_state="${STATE_DIR}/${model}.complete"
  if [[ -f "${complete_state}" ]]; then
    echo "[baseline-worker] skip completed model=${model}"
    continue
  fi
  input_list="${BASELINE_RUN_ROOT}/inputs/${model}.txt"
  expected_cases="$(wc -l < "${input_list}")"
  if [[ "${expected_cases}" -eq 0 ]]; then
    echo "model=${model}" > "${complete_state}"
    continue
  fi
  task_log="${LOG_DIR}/${model}.log"
  validation="${BASELINE_RUN_ROOT}/validations/${model}.json"
  case "${model}" in
    wan_lora|xssc)
      config_root="${BASELINE_OUTPUT_BASE}/${model}/baseline"
      command=(
        env "${common_env[@]}"
        INPUT_LIST="${input_list}" OUTPUT_BASE="${BASELINE_OUTPUT_BASE}"
        CFG_SCALE="${CFG_SCALE}"
        WAN_ROOT="${WAN_ROOT}" WAN_LORA_ROOT="${WAN_LORA_ROOT}"
        XSSC_WEIGHTS_ROOT="${XSSC_WEIGHTS_ROOT}"
        XSSC_ROOT="${XSSC_ROOT}" XSSC_CONFIG="${XSSC_CONFIG}"
        XSSC_CHECKPOINT="${XSSC_CHECKPOINT}"
        bash "${SCRIPT_DIR}/run_physiciq_one.sh"
        "${model}" baseline none "${GPU_ID}"
      )
      ;;
    physrvg)
      config_root="${BASELINE_OUTPUT_BASE}/PhyRVG/baseline"
      command=(
        env "${common_env[@]}"
        INPUT_LIST="${input_list}"
        OUTPUT_BASE="${BASELINE_OUTPUT_BASE}/PhyRVG"
        GUIDANCE_SCALE="${GUIDANCE_SCALE}" DO_CFG="${PHYSRVG_DO_CFG}"
        PHYSRVG_ROOT="${PHYSRVG_ROOT}" MODEL_ID="${PHYSRVG_MODEL_ID}"
        DIT_CHECKPOINT="${PHYSRVG_DIT_CHECKPOINT}"
        LORA_CHECKPOINT="${PHYSRVG_LORA_CHECKPOINT}"
        bash "${SCRIPT_DIR}/run_physrvg_physiciq_one.sh"
        baseline none "${GPU_ID}"
      )
      ;;
    *)
      echo "[baseline-worker] unsupported model=${model}"
      exit 2
      ;;
  esac

  {
    echo "model=${model}"
    echo "gpu=${GPU_ID}"
    echo "expected_cases=${expected_cases}"
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "${task_log}"
  echo "[baseline-worker] run model=${model} cases=${expected_cases} gpu=${GPU_ID}"
  status=0
  "${command[@]}" >> "${task_log}" 2>&1 || status=$?
  if [[ "${status}" -eq 0 ]]; then
    "${PYTHON}" "${VERIFY}" \
      --config-root "${config_root}" \
      --input-list "${input_list}" \
      --model "${model}" \
      --mode baseline \
      --block none \
      --expected-cases "${expected_cases}" \
      --output "${validation}" >> "${task_log}" 2>&1 || status=$?
  fi
  if [[ "${status}" -ne 0 ]]; then
    printf 'model=%s\nstatus=%s\n' "${model}" "${status}" \
      > "${STATE_DIR}/${model}.failed"
    exit "${status}"
  fi
  printf 'model=%s\ngpu=%s\ncases=%s\n' \
    "${model}" "${GPU_ID}" "${expected_cases}" > "${complete_state}"
  rm -f "${STATE_DIR}/${model}.failed"
  python3 "${GALLERY_SCRIPT}" \
    --root "${BASELINE_OUTPUT_BASE}" --build-only
done

printf 'worker=%s\ngpu=%s\nfinished_utc=%s\n' \
  "${WORKER_NAME}" "${GPU_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${STATE_DIR}/${WORKER_NAME}.worker.complete"

