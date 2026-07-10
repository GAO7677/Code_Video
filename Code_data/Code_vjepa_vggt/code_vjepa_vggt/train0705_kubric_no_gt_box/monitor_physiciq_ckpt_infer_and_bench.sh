#!/usr/bin/env bash
set -euo pipefail

# Watches a checkpoints directory for new step-* weights.
# When a new checkpoint lands and GPU 7 is idle, it runs physicIQ batch inference,
# appends the generated folder to AAAevalphysiq.txt, and launches bench.sh.
#
# Typical foreground usage:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/monitor_physiciq_ckpt_infer_and_bench.sh
#
# Typical tmux usage:
# tmux new-session -s physiciq_ckpt_watch \
#   "bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/monitor_physiciq_ckpt_infer_and_bench.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_INDEX="${GPU_INDEX:-7}"
GPU_PAIR="${GPU_PAIR:-7,7}"
AUTO_SPLIT_INPUT="${AUTO_SPLIT_INPUT:-1}"
TEST_JSON_TXT="${TEST_JSON_TXT:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/train_stage1b_kubric0708_regdiag_resume3500_20260710/checkpoints/step-004500/}"
METHOD_NAME="${METHOD_NAME:-train_stage1b_kubric0708_regdiag_step004500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_kubric0708}"
OUTPUT_FRAMES="${OUTPUT_FRAMES:-49}"
CTX="${CTX:-8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-896}"
NEG_TAG="${NEG_TAG:-defaultnegprompt}"
POLL_SECONDS="${POLL_SECONDS:-120}"
AAA_EVAL_TXT="${AAA_EVAL_TXT:-${SCRIPT_DIR}/AAAevalphysiq.txt}"
STATE_ROOT="${STATE_ROOT:-/data/gaoya/agent-data/cache/physiciq_ckpt_monitor}"
RUN_INFER_SH="${RUN_INFER_SH:-${SCRIPT_DIR}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh}"
BENCH_SH="${BENCH_SH:-${SCRIPT_DIR}/bench.sh}"

mkdir -p "${STATE_ROOT}"
LOCK_DIR="${STATE_ROOT}/lock_gpu${GPU_INDEX}"
PROCESSED_FILE="${STATE_ROOT}/processed_gpu${GPU_INDEX}.txt"
FAILED_FILE="${STATE_ROOT}/failed_gpu${GPU_INDEX}.txt"
touch "${PROCESSED_FILE}" "${FAILED_FILE}" "${AAA_EVAL_TXT}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

trim_trailing_slash() {
  local value="$1"
  value="${value%/}"
  echo "${value}"
}

normalize_ckpt_method_name() {
  local name="$1"
  echo "${name}" | sed -E 's/^[A-Za-z]+[0-9]+_//'
}

resolve_watch_root() {
  local raw_root
  local base_name
  raw_root="$(trim_trailing_slash "${WEIGHTS_ROOT}")"
  base_name="$(basename "${raw_root}")"
  if [[ "${base_name}" =~ ^step-[0-9]+$ ]]; then
    dirname "${raw_root}"
  else
    echo "${raw_root}"
  fi
}

resolve_method_template_prefix() {
  local raw_name="$1"
  if [[ "${raw_name}" =~ ^(.*_step)-?[0-9]+$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  echo "${raw_name}"
}

resolve_method_name_for_step() {
  local step_dir_name="$1"
  local prefix="$2"
  local digits
  digits="$(echo "${step_dir_name}" | sed -E 's/^step-//')"
  echo "${prefix}${digits}"
}

resolve_expected_output_dir() {
  local step_dir="$1"
  local step_name
  local checkpoints_root
  local train_root_name
  local method_root
  step_name="$(basename "${step_dir}")"
  checkpoints_root="$(dirname "${step_dir}")"
  train_root_name="$(basename "$(dirname "${checkpoints_root}")")"
  method_root="$(normalize_ckpt_method_name "${train_root_name}")"
  printf "%s/%s_%s_steps%02d_%sx%s_ctx%02d_%02df_%s" \
    "${OUTPUT_ROOT}" \
    "${method_root}" \
    "${step_name}" \
    "${NUM_INFERENCE_STEPS}" \
    "${HEIGHT}" \
    "${WIDTH}" \
    "${CTX}" \
    "${OUTPUT_FRAMES}" \
    "${NEG_TAG}"
}

gpu_uuid_for_index() {
  local target_index="$1"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F',' -v idx="${target_index}" '
        {
          gsub(/^[ \t]+|[ \t]+$/, "", $1)
          gsub(/^[ \t]+|[ \t]+$/, "", $2)
          if ($1 == idx) {
            print $2
            exit
          }
        }
      '
}

gpu_is_idle() {
  local target_index="$1"
  local gpu_uuid
  gpu_uuid="$(gpu_uuid_for_index "${target_index}")"
  if [ -z "${gpu_uuid}" ]; then
    log "WARN: failed to resolve GPU index ${target_index}"
    return 1
  fi
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | awk -F',' -v uuid="${gpu_uuid}" '
        {
          gsub(/^[ \t]+|[ \t]+$/, "", $1)
          if ($1 == uuid) {
            found = 1
          }
        }
        END { exit(found ? 0 : 1) }
      '; then
    return 1
  fi
  return 0
}

step_ready() {
  local step_dir="$1"
  [ -f "${step_dir}/checkpoint.safetensors" ] && [ -f "${step_dir}/training_state.pt" ]
}

already_processed() {
  local step_name="$1"
  rg -Fxq "${step_name}" "${PROCESSED_FILE}"
}

mark_processed() {
  local step_name="$1"
  echo "${step_name}" >> "${PROCESSED_FILE}"
}

mark_failed() {
  local step_name="$1"
  local reason="$2"
  echo "${step_name}	${reason}	$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${FAILED_FILE}"
}

append_eval_path_if_missing() {
  local output_dir="$1"
  if ! rg -Fxq "${output_dir}" "${AAA_EVAL_TXT}"; then
    echo "${output_dir}" >> "${AAA_EVAL_TXT}"
    log "appended eval path: ${output_dir}"
  else
    log "eval path already present: ${output_dir}"
  fi
}

run_infer_for_step() {
  local step_dir="$1"
  local step_name
  local method_template_prefix
  local method_name_for_step
  step_name="$(basename "${step_dir}")"
  method_template_prefix="$(resolve_method_template_prefix "${METHOD_NAME}")"
  method_name_for_step="$(resolve_method_name_for_step "${step_name}" "${method_template_prefix}")"

  log "start infer for ${step_name}"
  env \
    GPU_PAIR="${GPU_PAIR}" \
    AUTO_SPLIT_INPUT="${AUTO_SPLIT_INPUT}" \
    TEST_JSON_TXT="${TEST_JSON_TXT}" \
    WEIGHTS_ROOT="${step_dir}" \
    METHOD_NAME="${method_name_for_step}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    OUTPUT_FRAMES="${OUTPUT_FRAMES}" \
    CTX="${CTX}" \
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
    HEIGHT="${HEIGHT}" \
    WIDTH="${WIDTH}" \
    bash "${RUN_INFER_SH}"
}

run_bench() {
  log "start bench on gpu${GPU_INDEX}"
  CUDA_VISIBLE_DEVICES="${GPU_INDEX}" bash "${BENCH_SH}" "${AAA_EVAL_TXT}"
  log "bench completed"
}

process_step() {
  local step_dir="$1"
  local step_name
  local expected_output_dir
  step_name="$(basename "${step_dir}")"
  expected_output_dir="$(resolve_expected_output_dir "${step_dir}")"

  if [ -d "${expected_output_dir}" ] && [ -f "${expected_output_dir}/result.json" ]; then
    log "reuse existing infer output for ${step_name}: ${expected_output_dir}"
  else
    run_infer_for_step "${step_dir}"
  fi

  if [ ! -d "${expected_output_dir}" ]; then
    mark_failed "${step_name}" "missing_output_dir:${expected_output_dir}"
    log "ERROR: expected output dir missing after infer: ${expected_output_dir}"
    return 1
  fi

  append_eval_path_if_missing "${expected_output_dir}"
  run_bench
  mark_processed "${step_name}"
  log "finished ${step_name}"
}

collect_candidate_steps() {
  local watch_root="$1"
  find "${watch_root}" -maxdepth 1 -mindepth 1 -type d -name 'step-*' | sort -V
}

main() {
  local watch_root
  local step_dir
  local step_name

  watch_root="$(resolve_watch_root)"
  if [ ! -d "${watch_root}" ]; then
    echo "ERROR: watch root does not exist: ${watch_root}" >&2
    exit 1
  fi
  if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "ERROR: another monitor instance appears to be running: ${LOCK_DIR}" >&2
    exit 1
  fi
  trap 'rmdir "${LOCK_DIR}" >/dev/null 2>&1 || true' EXIT

  log "watch_root=${watch_root}"
  log "gpu_index=${GPU_INDEX} gpu_pair=${GPU_PAIR}"
  log "output_root=${OUTPUT_ROOT}"
  log "eval_txt=${AAA_EVAL_TXT}"

  while true; do
    while IFS= read -r step_dir; do
      [ -n "${step_dir}" ] || continue
      step_name="$(basename "${step_dir}")"
      if already_processed "${step_name}"; then
        continue
      fi
      if ! step_ready "${step_dir}"; then
        continue
      fi
      if ! gpu_is_idle "${GPU_INDEX}"; then
        log "gpu${GPU_INDEX} busy, defer ${step_name}"
        break
      fi
      if ! process_step "${step_dir}"; then
        log "ERROR: pipeline failed for ${step_name}; will retry on next poll"
        sleep "${POLL_SECONDS}"
      fi
    done < <(collect_candidate_steps "${watch_root}")
    sleep "${POLL_SECONDS}"
  done
}

main "$@"
