#!/usr/bin/env bash
set -euo pipefail

# Run:
# tmux new-session -d -s watch_xssc_pybullet0717_test5_gpu4 -- \
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/watch_xssc_pybullet0717_test5_checkpoints.sh

RUN_ROOT="${RUN_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots_pybullet0717/wan22_5b_pybullet0717_49f_ctx08_20260723T105506Z}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/test_5}"
RUN_ID="$(basename "${RUN_ROOT}")"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_ROOT}/pybullet0717_xssc_context_slots_${RUN_ID}}"
GPU_ID="${GPU_ID:-4}"
TEST_LIST="${TEST_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
RUN_INFER="${RUN_INFER:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots_pybullet0717_test5.sh}"
VALIDATE_SCRIPT="${VALIDATE_SCRIPT:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/validate_xssc_inference.py}"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_SECONDS="${STABLE_SECONDS:-20}"
MAX_RETRIES="${MAX_RETRIES:-3}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-2500}"
IDLE_UTIL_PERCENT="${IDLE_UTIL_PERCENT:-20}"
WAIT_FOR_IDLE_GPU="${WAIT_FOR_IDLE_GPU:-1}"

STATE_DIR="${WATCHER_STATE_DIR:-${OUTPUT_BASE}/_watcher_state}"
DONE_DIR="${STATE_DIR}/done"
LOG_DIR="${STATE_DIR}/logs"
mkdir -p "${DONE_DIR}" "${LOG_DIR}" "${OUTPUT_BASE}"

if [[ ! -s "${TEST_LIST}" ]]; then
  echo "TEST_LIST not found or empty: ${TEST_LIST}" >&2
  exit 2
fi
if [[ ! -x "${RUN_INFER}" ]]; then
  echo "RUN_INFER not executable: ${RUN_INFER}" >&2
  exit 2
fi
if [[ ! -f "${VALIDATE_SCRIPT}" ]]; then
  echo "VALIDATE_SCRIPT not found: ${VALIDATE_SCRIPT}" >&2
  exit 2
fi

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${STATE_DIR}/watcher.log"
}

wait_until_complete() {
  local ckpt_dir="$1"
  local ckpt_file="${ckpt_dir}/checkpoint.safetensors"
  local state_file="${ckpt_dir}/training_state.pt"
  [[ -s "${ckpt_file}" && -s "${state_file}" ]] || return 1
  local size1 size2
  size1="$(stat -c '%s' "${ckpt_file}")"
  sleep "${STABLE_SECONDS}"
  [[ -s "${ckpt_file}" && -s "${state_file}" ]] || return 1
  size2="$(stat -c '%s' "${ckpt_file}")"
  [[ "${size1}" == "${size2}" ]]
}

wait_for_idle_gpu() {
  if [[ "${WAIT_FOR_IDLE_GPU}" != "1" ]]; then
    return 0
  fi
  while true; do
    local row memory util
    row="$(
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F, -v want="${GPU_ID}" '
            $1 + 0 == want {
              gsub(/ /, "", $2)
              gsub(/ /, "", $3)
              print $2 "," $3
              found = 1
            }
            END { if (!found) exit 1 }
          ' || true
    )"
    if [[ -z "${row}" ]]; then
      log "GPU ${GPU_ID} not visible; wait ${POLL_SECONDS}s"
      sleep "${POLL_SECONDS}"
      continue
    fi
    memory="${row%,*}"
    util="${row#*,}"
    if (( memory <= IDLE_MEMORY_MIB && util <= IDLE_UTIL_PERCENT )); then
      return 0
    fi
    log "GPU ${GPU_ID} busy memory=${memory}MiB util=${util}%; wait ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

log "checkpoint_root=${CHECKPOINT_ROOT}"
log "output_base=${OUTPUT_BASE}"
log "gpu=${GPU_ID} test_list=${TEST_LIST} steps=${NUM_INFERENCE_STEPS}"
log "poll=${POLL_SECONDS}s stable=${STABLE_SECONDS}s max_retries=${MAX_RETRIES}"

while true; do
  if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    log "checkpoint root not found yet: ${CHECKPOINT_ROOT}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  mapfile -t checkpoints < <(
    find "${CHECKPOINT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'step-*' \
      | sort -V
  )

  if (( ${#checkpoints[@]} == 0 )); then
    log "waiting for checkpoints under ${CHECKPOINT_ROOT}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  for ckpt_dir in "${checkpoints[@]}"; do
    step_tag="$(basename "${ckpt_dir}")"
    done_marker="${DONE_DIR}/${step_tag}.done"
    running_marker="${DONE_DIR}/${step_tag}.running"
    failed_marker="${DONE_DIR}/${step_tag}.failed"
    attempts_file="${DONE_DIR}/${step_tag}.attempts"
    output_dir="${OUTPUT_BASE}/${step_tag}"
    log_file="${LOG_DIR}/${step_tag}.log"

    if [[ -e "${done_marker}" || -e "${running_marker}" ]]; then
      continue
    fi

    attempts=0
    if [[ -s "${attempts_file}" ]]; then
      attempts="$(<"${attempts_file}")"
    fi
    if (( attempts >= MAX_RETRIES )); then
      continue
    fi

    if ! wait_until_complete "${ckpt_dir}"; then
      log "${step_tag} not complete/stable yet"
      continue
    fi

    wait_for_idle_gpu
    exec 9>"/tmp/xssc_pybullet0717_test5_gpu_${GPU_ID}.lock"
    if ! flock -n 9; then
      log "GPU ${GPU_ID} inference lock is held; skip this poll"
      continue
    fi

    mkdir -p "${output_dir}"
    touch "${running_marker}"
    rm -f "${failed_marker}"
    log "start ${step_tag} on GPU ${GPU_ID} -> ${output_dir}"
    set +e
    TEST_LIST="${TEST_LIST}" \
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
    TRACE_ROOT="${output_dir}/numeric_traces/${step_tag}" \
    bash "${RUN_INFER}" "${ckpt_dir}" "${GPU_ID}" "${output_dir}" \
      > "${log_file}" 2>&1
    infer_status=$?
    if [[ "${infer_status}" -eq 0 ]]; then
      "${PYTHON}" "${VALIDATE_SCRIPT}" \
        --output-root "${output_dir}" \
        --input-json-list "${TEST_LIST}" \
        --report "${output_dir}/health_report.json" \
        >> "${log_file}" 2>&1
      validate_status=$?
    else
      validate_status=99
    fi
    set -e
    rm -f "${running_marker}"

    if [[ "${infer_status}" -eq 0 && "${validate_status}" -eq 0 ]]; then
      rm -f "${failed_marker}" "${attempts_file}"
      date -u '+%Y-%m-%dT%H:%M:%SZ' > "${done_marker}"
      touch "${output_dir}/.validated"
      log "done ${step_tag}"
    else
      attempts=$((attempts + 1))
      printf '%s\n' "${attempts}" > "${attempts_file}"
      {
        echo "infer_status=${infer_status}"
        echo "validate_status=${validate_status}"
        echo "attempts=${attempts}/${MAX_RETRIES}"
        echo "log=${log_file}"
        date -u '+%Y-%m-%dT%H:%M:%SZ'
      } > "${failed_marker}"
      log "failed ${step_tag}; attempt ${attempts}/${MAX_RETRIES}; see ${log_file}"
    fi
    flock -u 9
  done

  sleep "${POLL_SECONDS}"
done
