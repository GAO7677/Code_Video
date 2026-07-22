#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 CHECKPOINT_ROOT OUTPUT_ROOT [GPU_ID]" >&2
  exit 2
fi

CHECKPOINT_ROOT="$1"
OUTPUT_ROOT="$2"
GPU_ID="${3:-4}"

TEST_LIST="${TEST_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
RUN_INFER="${RUN_INFER:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots_dinov3.sh}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_SECONDS="${STABLE_SECONDS:-20}"
METHOD_PREFIX="${METHOD_PREFIX:-dinov3_movic_amg_step026000}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理}"

RUN_ID="$(basename "$(dirname "${CHECKPOINT_ROOT}")")"
STATE_DIR="${WATCHER_STATE_DIR:-${OUTPUT_ROOT}/_watcher_state/${METHOD_PREFIX}_${RUN_ID}}"
DONE_DIR="${STATE_DIR}/done"
LOG_DIR="${STATE_DIR}/logs"
mkdir -p "${DONE_DIR}" "${LOG_DIR}" "${OUTPUT_ROOT}"

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "CHECKPOINT_ROOT not found yet: ${CHECKPOINT_ROOT}; watcher will wait."
fi
if [[ ! -s "${TEST_LIST}" ]]; then
  echo "TEST_LIST not found or empty: ${TEST_LIST}" >&2
  exit 2
fi
if [[ ! -x "${RUN_INFER}" ]]; then
  echo "RUN_INFER not executable: ${RUN_INFER}" >&2
  exit 2
fi

checkpoint_step_num() {
  basename "$1" | awk -F- '{print $2 + 0}'
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

echo "[watcher] checkpoint_root=${CHECKPOINT_ROOT}"
echo "[watcher] output_root=${OUTPUT_ROOT}"
echo "[watcher] gpu=${GPU_ID} test_list=${TEST_LIST} steps=${NUM_INFERENCE_STEPS}"
echo "[watcher] poll=${POLL_SECONDS}s stable=${STABLE_SECONDS}s"

while true; do
  if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    sleep "${POLL_SECONDS}"
    continue
  fi

  mapfile -t checkpoints < <(
    find "${CHECKPOINT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'step-*' \
      | sort -V
  )

  for ckpt_dir in "${checkpoints[@]}"; do
    step_tag="$(basename "${ckpt_dir}")"
    done_marker="${DONE_DIR}/${step_tag}.done"
    running_marker="${DONE_DIR}/${step_tag}.running"
    failed_marker="${DONE_DIR}/${step_tag}.failed"
    if [[ -e "${done_marker}" || -e "${running_marker}" ]]; then
      continue
    fi
    if ! wait_until_complete "${ckpt_dir}"; then
      echo "[watcher] ${step_tag} not complete/stable yet"
      continue
    fi

    touch "${running_marker}"
    log_file="${LOG_DIR}/${step_tag}.log"
    method_name="${METHOD_PREFIX}_${step_tag}_steps${NUM_INFERENCE_STEPS}_512x896_ctx08_49f"
    echo "[watcher] start ${step_tag} -> ${method_name}"
    set +e
    TEST_LIST="${TEST_LIST}" \
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
    STEP_OUTPUT_DIR_NAME="${method_name}" \
    TRACE_ROOT="${OUTPUT_ROOT}/_numeric_traces/${step_tag}" \
    NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
    bash "${RUN_INFER}" "${ckpt_dir}" "${GPU_ID}" "${OUTPUT_ROOT}" \
      > "${log_file}" 2>&1
    status=$?
    set -e
    rm -f "${running_marker}"
    if [[ "${status}" -eq 0 ]]; then
      date -u '+%Y-%m-%dT%H:%M:%SZ' > "${done_marker}"
      echo "[watcher] done ${step_tag}"
    else
      {
        echo "status=${status}"
        echo "log=${log_file}"
        date -u '+%Y-%m-%dT%H:%M:%SZ'
      } > "${failed_marker}"
      echo "[watcher] failed ${step_tag}; see ${log_file}" >&2
    fi
  done

  sleep "${POLL_SECONDS}"
done
