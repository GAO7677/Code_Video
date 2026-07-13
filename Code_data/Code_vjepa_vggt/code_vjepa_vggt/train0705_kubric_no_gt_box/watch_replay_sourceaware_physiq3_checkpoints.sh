#!/usr/bin/env bash
set -u

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
INFER_WRAPPER="${BASE}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the training run checkpoints directory}"
INPUT_LIST="${INPUT_LIST:-/data/gaoya/agent-data/outputs/replay_preserve_step300_physicIQ_025_026_three_cases_20260713/input_jsons.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a temporary validation output directory under /data/gaoya}"
MODEL_PREFIX="${MODEL_PREFIX:-replay_sourceaware_physiq3}"
GPU_PAIR="${GPU_PAIR:-6,6}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MIN_CHECKPOINT_AGE_SECONDS="${MIN_CHECKPOINT_AGE_SECONDS:-90}"
TMP_ROOT="${TMP_ROOT:-/data/gaoya/agent-data/cache/t/replay_physiq3_watch}"
STATE_ROOT="${STATE_ROOT:-${OUTPUT_ROOT}/_watch_state}"
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
  [ -f "${result_json}" ] || return 1
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python - "${result_json}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
complete = (
    int(result.get("num_total", -1)) == 3
    and int(result.get("num_success", -1)) == 3
    and int(result.get("num_failed", -1)) == 0
)
raise SystemExit(0 if complete else 1)
PY
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
      log "infer incomplete ${step_name}: wrapper exited 0 without a 3/3 result"
    fi
  else
    log "infer failed ${step_name}: it will be retried after ${POLL_SECONDS}s"
  fi
}

if [ "$(wc -l < "${INPUT_LIST}")" -ne 3 ]; then
  echo "ERROR: INPUT_LIST must contain exactly three cases: ${INPUT_LIST}" >&2
  exit 1
fi

log "watcher start checkpoint_root=${CHECKPOINT_ROOT} gpu_pair=${GPU_PAIR} input_list=${INPUT_LIST}"
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
