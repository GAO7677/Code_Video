#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must point to an xSSC training run" >&2
  exit 2
fi

PROJECT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
INPUT_JSON_LIST="${INPUT_JSON_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/agent-data/outputs/xssc_checkpoint_inference/$(basename "${RUN_ROOT}")}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-1200}"
IDLE_UTIL_PERCENT="${IDLE_UTIL_PERCENT:-10}"
EXCLUDE_GPU_IDS=",${EXCLUDE_GPU_IDS:-0,2,3,4,5},"

mkdir -p "${OUTPUT_BASE}"

select_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | while IFS=',' read -r gpu memory util; do
      gpu="${gpu//[[:space:]]/}"
      memory="${memory//[[:space:]]/}"
      util="${util//[[:space:]]/}"
      [[ "${EXCLUDE_GPU_IDS}" == *",${gpu},"* ]] && continue
      if (( memory <= IDLE_MEMORY_MIB && util <= IDLE_UTIL_PERCENT )); then
        echo "${gpu}"
        return 0
      fi
    done
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
    [[ -f "${output_root}/.validated" || -f "${output_root}/.failed" ]] && continue

    gpu="$(select_idle_gpu || true)"
    if [[ -z "${gpu}" ]]; then
      printf '%s no idle inference GPU available\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      break
    fi
    exec 9>"/tmp/xssc_inference_gpu_${gpu}.lock"
    if ! flock -n 9; then
      continue
    fi

    mkdir -p "${output_root}"
    printf '%s starting %s on physical GPU %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${step_name}" "${gpu}" \
      | tee -a "${OUTPUT_BASE}/watcher.log"
    if TEST_LIST="${INPUT_JSON_LIST}" \
      NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}" \
      bash "${PROJECT}/run_infer_xssc_context_slots.sh" \
        "${checkpoint_dir}" "${gpu}" "${output_root}" \
        2>&1 | tee "${output_root}/inference.log" && \
      "${PYTHON}" "${PROJECT}/validate_xssc_inference.py" \
        --output-root "${output_root}" \
        --input-json-list "${INPUT_JSON_LIST}" \
        --report "${output_root}/health_report.json" \
        2>&1 | tee -a "${output_root}/inference.log"; then
      touch "${output_root}/.validated"
      status=completed
    else
      touch "${output_root}/.failed"
      status=failed
    fi
    printf '%s %s %s on GPU %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "${step_name}" "${gpu}" \
      | tee -a "${OUTPUT_BASE}/watcher.log"
    flock -u 9
  done < <(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' 2>/dev/null | sort -V)

  if (( found_ready == 0 )); then
    printf '%s waiting for checkpoints under %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${CHECKPOINT_ROOT}"
  fi
  sleep "${POLL_SECONDS}"
done
