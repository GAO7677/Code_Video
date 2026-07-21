#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must point to a center-crop pooled xSSC training run" >&2
  exit 2
fi

PROJECT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
INPUT_JSON_LIST="${INPUT_JSON_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/test_5/$(basename "${RUN_ROOT}")}"
VIEWER_ROOT="${VIEWER_ROOT:-$(dirname "${OUTPUT_BASE}")}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TARGET_GPU="${TARGET_GPU:-6}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-1200}"
IDLE_UTIL_PERCENT="${IDLE_UTIL_PERCENT:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"

mkdir -p "${OUTPUT_BASE}"

target_gpu_is_idle() {
  local line gpu memory util
  line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' -v target="${TARGET_GPU}" '
      {
        gsub(/[[:space:]]/, "", $1);
        if ($1 == target) {
          gsub(/[[:space:]]/, "", $2);
          gsub(/[[:space:]]/, "", $3);
          print $1 "," $2 "," $3;
          exit 0;
        }
      }')"
  [[ -n "${line}" ]] || return 1
  IFS=',' read -r gpu memory util <<< "${line}"
  if (( memory <= IDLE_MEMORY_MIB && util <= IDLE_UTIL_PERCENT )); then
    return 0
  fi
  printf '%s GPU%s busy: memory=%sMiB util=%s%%; waiting for <=%sMiB and <=%s%%\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TARGET_GPU}" "${memory}" "${util}" \
    "${IDLE_MEMORY_MIB}" "${IDLE_UTIL_PERCENT}" \
    | tee -a "${OUTPUT_BASE}/watcher.log"
  return 1
}

while true; do
  found_ready=0
  while IFS= read -r checkpoint_dir; do
    [[ -n "${checkpoint_dir}" ]] || continue
    [[ -s "${checkpoint_dir}/checkpoint.safetensors" ]] || continue
    [[ -s "${checkpoint_dir}/training_state.pt" ]] || continue
    found_ready=1

    step_name="$(basename "${checkpoint_dir}")"
    output_root="${OUTPUT_BASE}/${step_name}"
    [[ -f "${output_root}/.validated" ]] && continue

    attempts_file="${output_root}/attempts.txt"
    attempts=0
    if [[ -s "${attempts_file}" ]]; then
      attempts="$(<"${attempts_file}")"
    fi
    if (( attempts >= MAX_RETRIES )); then
      continue
    fi

    if ! target_gpu_is_idle; then
      break
    fi

    exec 9>"/tmp/xssc_centercrop_pooled_inference_gpu_${TARGET_GPU}.lock"
    if ! flock -n 9; then
      printf '%s GPU%s inference lock is held; waiting\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TARGET_GPU}" \
        | tee -a "${OUTPUT_BASE}/watcher.log"
      break
    fi

    mkdir -p "${output_root}"
    printf '%s starting center-crop pooled %s on physical GPU %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${step_name}" "${TARGET_GPU}" \
      | tee -a "${OUTPUT_BASE}/watcher.log"

    if TEST_LIST="${INPUT_JSON_LIST}" \
      NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
      STEP_OUTPUT_DIR_NAME="${step_name}" \
      XSSC_TRAIN_CROP_MODE=center \
      XSSC_EVAL_CROP_MODE=center \
      bash "${PROJECT}/run_infer_xssc_randomcrop_pooled_slots.sh" \
        "${checkpoint_dir}" "${TARGET_GPU}" "${output_root}" \
        2>&1 | tee "${output_root}/inference.log" && \
      "${PYTHON}" "${PROJECT}/validate_xssc_inference.py" \
        --output-root "${output_root}/${step_name}" \
        --input-json-list "${INPUT_JSON_LIST}" \
        --report "${output_root}/health_report.json" \
        2>&1 | tee -a "${output_root}/inference.log"; then
      "${PYTHON}" "${PROJECT}/build_test5_comparison_viewer.py" \
        --root "${VIEWER_ROOT}" \
        2>&1 | tee -a "${output_root}/inference.log"
      rm -f "${output_root}/.failed" "${attempts_file}"
      touch "${output_root}/.validated"
      status=completed
    else
      attempts=$((attempts + 1))
      printf '%s\n' "${attempts}" > "${attempts_file}"
      touch "${output_root}/.failed"
      status="failed attempt ${attempts}/${MAX_RETRIES}"
    fi

    printf '%s %s center-crop pooled %s on GPU %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "${step_name}" "${TARGET_GPU}" \
      | tee -a "${OUTPUT_BASE}/watcher.log"
    flock -u 9
  done < <(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' 2>/dev/null | sort -V)

  if (( found_ready == 0 )); then
    printf '%s waiting for checkpoints under %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${CHECKPOINT_ROOT}" \
      | tee -a "${OUTPUT_BASE}/watcher.log"
  fi
  sleep "${POLL_SECONDS}"
done
