#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must point to a Scheme A all-frame oracle xSSC training run" >&2
  exit 2
fi

PROJECT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
INPUT_JSON_LIST="${INPUT_JSON_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/test_5/train_xssc_allframe_oracle_slots}"
VIEWER_ROOT="${VIEWER_ROOT:-$(dirname "${OUTPUT_BASE}")}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TARGET_GPU="${TARGET_GPU:-6}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-1200}"
IDLE_UTIL_PERCENT="${IDLE_UTIL_PERCENT:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-8}"
NUM_FRAMES="${NUM_FRAMES:-49}"
XSSC_ORACLE_VIDEO_FRAMES="${XSSC_ORACLE_VIDEO_FRAMES:-49}"
XSSC_VAE_TEMPORAL_STRIDE="${XSSC_VAE_TEMPORAL_STRIDE:-4}"

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

    exec 9>"/tmp/xssc_allframe_oracle_inference_gpu_${TARGET_GPU}.lock"
    if ! flock -n 9; then
      printf '%s GPU%s all-frame oracle inference lock is held; waiting\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TARGET_GPU}" \
        | tee -a "${OUTPUT_BASE}/watcher.log"
      break
    fi

    mkdir -p "${output_root}" "${output_root}/numeric_traces"
    printf '%s starting all-frame oracle %s on physical GPU %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${step_name}" "${TARGET_GPU}" \
      | tee -a "${OUTPUT_BASE}/watcher.log"

    if PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
      CUDA_VISIBLE_DEVICES="${TARGET_GPU}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      HF_HOME=/data/gaoya/agent-data/cache/xssc_wan/huggingface \
      TORCH_HOME=/data/gaoya/agent-data/cache/xssc_wan/torch \
      XDG_CACHE_HOME=/data/gaoya/agent-data/cache/xssc_wan/xdg \
      "${PYTHON}" "${PROJECT}/batch_infer_xssc_allframe_oracle_slots.py" \
        --weights-root "${checkpoint_dir}" \
        --input-json-list-path "${INPUT_JSON_LIST}" \
        --model-name train_xssc_allframe_oracle_slots \
        --output-root "${output_root}" \
        --step-output-dir-name "${step_name}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --context-frames "${CONTEXT_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --xssc-oracle-video-frames "${XSSC_ORACLE_VIDEO_FRAMES}" \
        --xssc-vae-temporal-stride "${XSSC_VAE_TEMPORAL_STRIDE}" \
        --xssc-oracle-sampling-mode prefix \
        --xssc-oracle-video-resize-mode cover_crop \
        --xssc-preprocess-mode center_crop \
        --force \
        2>&1 | tee "${output_root}/inference.log" && \
      "${PYTHON}" "${PROJECT}/validate_xssc_inference.py" \
        --output-root "${output_root}" \
        --input-json-list "${INPUT_JSON_LIST}" \
        --expected-frames "${NUM_FRAMES}" \
        --expected-height 512 \
        --expected-width 896 \
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

    printf '%s %s all-frame oracle %s on GPU %s\n' \
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
