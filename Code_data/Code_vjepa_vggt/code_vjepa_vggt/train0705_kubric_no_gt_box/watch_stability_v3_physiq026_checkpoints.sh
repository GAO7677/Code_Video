#!/usr/bin/env bash
set -u

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
INFER_WRAPPER="${BASE}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708_stability_v3_from_scratch_20260711T144000Z/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_kubric0708}"
INPUT_LIST="${INPUT_LIST:-/data/gaoya/agent-data/outputs/query_prior_compare_20260710/physicIQ_026_mask_vs_boxuniform/ablllllll/_single_case_input_json.txt}"
MODEL_PREFIX="${MODEL_PREFIX:-train_stage1b_kubric0708_stability_v3_from_scratch_20260711T144000Z}"
GPU_PAIR="${GPU_PAIR:-6,6}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MIN_CHECKPOINT_AGE_SECONDS="${MIN_CHECKPOINT_AGE_SECONDS:-90}"
TMP_ROOT="${TMP_ROOT:-/data/gaoya/agent-data/cache/tmp/watch_stability_v3_physiq026}"
STATE_ROOT="${STATE_ROOT:-/data/gaoya/agent-data/outputs/watch_stability_v3_physiq026_20260711}"
WATCH_LOG="${STATE_ROOT}/watch.log"
LOCK_DIR="${STATE_ROOT}/watch.lock"

mkdir -p "${OUTPUT_ROOT}" "${TMP_ROOT}" "${STATE_ROOT}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: watcher already running or stale lock exists: ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${WATCH_LOG}"
}

result_is_complete() {
  local result_json="$1"
  [ -f "${result_json}" ] && \
    rg -q '"num_success"[[:space:]]*:[[:space:]]*1' "${result_json}" && \
    rg -q '"num_failed"[[:space:]]*:[[:space:]]*0' "${result_json}"
}

checkpoint_is_ready() {
  local step_dir="$1"
  local checkpoint_file="${step_dir}/checkpoint.safetensors"
  local training_state="${step_dir}/training_state.pt"
  local now mtime age
  [ -s "${checkpoint_file}" ] && [ -s "${training_state}" ] || return 1
  now="$(date +%s)"
  mtime="$(stat -c '%Y' "${checkpoint_file}")"
  age=$((now - mtime))
  [ "${age}" -ge "${MIN_CHECKPOINT_AGE_SECONDS}" ]
}

gpu_is_available() {
  local gpu_index="${GPU_PAIR%%,*}"
  local memory_used
  memory_used="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  [ -n "${memory_used}" ] && [ "${memory_used}" -lt 2048 ]
}

run_checkpoint() {
  local step_dir="$1"
  local step_name model_name output_dir tmp_dir
  step_name="$(basename "${step_dir}")"
  model_name="${MODEL_PREFIX}_${step_name}"
  output_dir="${OUTPUT_ROOT}/${model_name}_steps40_512x896_ctx08_49f_defaultnegprompt"
  tmp_dir="${TMP_ROOT}/${step_name}"

  if result_is_complete "${output_dir}/result.json"; then
    log "skip completed ${step_name}: ${output_dir}"
    return 0
  fi
  if ! gpu_is_available; then
    log "defer ${step_name}: GPU pair ${GPU_PAIR} is busy"
    return 0
  fi

  mkdir -p "${tmp_dir}"
  log "infer start ${step_name}: checkpoint=${step_dir} output=${output_dir}"
  if env \
    GPU_PAIR="${GPU_PAIR}" \
    TEST_JSON_TXT="${INPUT_LIST}" \
    WEIGHTS_ROOT="${step_dir}" \
    METHOD_NAME="${model_name}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    OUTPUT_FRAMES=49 \
    CTX=8 \
    NUM_INFERENCE_STEPS=40 \
    CFG_SCALE=5.0 \
    SEED=42 \
    COMPACT_OBJECT_CONTEXT_SLOTS=1 \
    OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO=3.0 \
    OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.30 \
    OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
    FORCE=1 \
    TMPDIR="${tmp_dir}" TMP="${tmp_dir}" TEMP="${tmp_dir}" \
    bash "${INFER_WRAPPER}" >>"${WATCH_LOG}" 2>&1; then
    if result_is_complete "${output_dir}/result.json"; then
      log "infer success ${step_name}: ${output_dir}"
    else
      log "infer incomplete ${step_name}: wrapper exited 0 without successful result.json"
    fi
  else
    log "infer failed ${step_name}: it will be retried after ${POLL_SECONDS}s"
  fi
}

log "watcher start checkpoint_root=${CHECKPOINT_ROOT} gpu_pair=${GPU_PAIR}"
while true; do
  found=0
  while IFS= read -r step_dir; do
    [ -n "${step_dir}" ] || continue
    found=1
    if checkpoint_is_ready "${step_dir}"; then
      run_checkpoint "${step_dir}"
    fi
  done < <(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V)
  if [ "${found}" -eq 0 ]; then
    log "waiting: no checkpoints under ${CHECKPOINT_ROOT}"
  fi
  sleep "${POLL_SECONDS}"
done
