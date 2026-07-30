#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/watch_formal_step500_and_infer.sh
set -euo pipefail

RUN_TAG="${RUN_TAG:-formal_20260729T184553Z}"
STEP_TAG="${STEP_TAG:-step-000500}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABILITY_SECONDS="${STABILITY_SECONDS:-30}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-8000}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-2}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 3 5}"

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_BASE=/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora
CONTROL_ROOT="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_step500_control/${RUN_TAG}"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_formal_step500/${RUN_TAG}"
TEST_LIST="${TEST_LIST:-/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_control/20260729T184553Z/test_5_first_case.txt}"

EXPERIMENT_NAMES=(
  object_only_gpu1_formal
  full_sa_gpu2_formal
  same_frame_s_head_full59_gpu6_formal
  common_t_head_full70_gpu7_formal
)
RUN_DIR_NAMES=(
  object_only_gpu1_formal
  full_sa_gpu2_formal
  same_frame_s_head_full59_gpu6_formal
  common_t_head_full70_gpu7_formal
)

mkdir -p "${CONTROL_ROOT}/logs" "${OUTPUT_ROOT}"

timestamp() {
  date -u '+%Y-%m-%d %H:%M:%S UTC'
}

checkpoint_dir_for() {
  local run_dir_name="$1"
  printf '%s/%s/%s/checkpoints/%s\n' \
    "${CHECKPOINT_BASE}" "${run_dir_name}" "${RUN_TAG}" "${STEP_TAG}"
}

checkpoint_complete() {
  local checkpoint_dir="$1"
  [[ -s "${checkpoint_dir}/checkpoint.safetensors" ]] &&
    [[ -s "${checkpoint_dir}/training_state.pt" ]]
}

checkpoint_size_signature() {
  local checkpoint_dir="$1"
  stat -c '%n:%s' \
    "${checkpoint_dir}/checkpoint.safetensors" \
    "${checkpoint_dir}/training_state.pt"
}

wait_for_checkpoints() {
  local all_ready checkpoint_dir run_dir_name
  while true; do
    all_ready=1
    for run_dir_name in "${RUN_DIR_NAMES[@]}"; do
      checkpoint_dir="$(checkpoint_dir_for "${run_dir_name}")"
      if checkpoint_complete "${checkpoint_dir}"; then
        printf '[%s] ready: %s\n' "$(timestamp)" "${checkpoint_dir}"
      else
        printf '[%s] waiting: %s\n' "$(timestamp)" "${checkpoint_dir}"
        all_ready=0
      fi
    done
    (( all_ready == 1 )) && break
    sleep "${POLL_SECONDS}"
  done
}

wait_for_stable_checkpoints() {
  local after before checkpoint_dir run_dir_name
  while true; do
    before=""
    for run_dir_name in "${RUN_DIR_NAMES[@]}"; do
      checkpoint_dir="$(checkpoint_dir_for "${run_dir_name}")"
      before+="$(checkpoint_size_signature "${checkpoint_dir}")"$'\n'
    done
    printf '[%s] checking checkpoint size stability for %ss\n' \
      "$(timestamp)" "${STABILITY_SECONDS}"
    sleep "${STABILITY_SECONDS}"
    after=""
    for run_dir_name in "${RUN_DIR_NAMES[@]}"; do
      checkpoint_dir="$(checkpoint_dir_for "${run_dir_name}")"
      if ! checkpoint_complete "${checkpoint_dir}"; then
        after="incomplete"
        break
      fi
      after+="$(checkpoint_size_signature "${checkpoint_dir}")"$'\n'
    done
    if [[ "${before}" == "${after}" ]]; then
      printf '[%s] all checkpoint files are stable\n' "$(timestamp)"
      return
    fi
    printf '[%s] checkpoint sizes changed; checking again\n' "$(timestamp)"
  done
}

select_available_gpu() {
  local gpu memory_used
  for gpu in ${GPU_CANDIDATES}; do
    if [[ "${gpu}" == "4" ]]; then
      continue
    fi
    memory_used="$(
      nvidia-smi -i "${gpu}" --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' '
    )"
    if [[ "${memory_used}" =~ ^[0-9]+$ ]] &&
      (( memory_used <= MAX_GPU_MEMORY_USED_MIB )); then
      printf '%s\n' "${gpu}"
      return 0
    fi
  done
  return 1
}

wait_for_available_gpu() {
  local gpu
  while true; do
    if gpu="$(select_available_gpu)"; then
      printf '%s\n' "${gpu}"
      return
    fi
    printf '[%s] no inference GPU below %s MiB; candidates: %s\n' \
      "$(timestamp)" "${MAX_GPU_MEMORY_USED_MIB}" "${GPU_CANDIDATES}" >&2
    sleep "${GPU_POLL_SECONDS}"
  done
}

run_inference() {
  local checkpoint_dir experiment_name gpu index log_path
  gpu="$1"
  for index in "${!EXPERIMENT_NAMES[@]}"; do
    experiment_name="${EXPERIMENT_NAMES[${index}]}"
    checkpoint_dir="$(checkpoint_dir_for "${RUN_DIR_NAMES[${index}]}")"
    log_path="${CONTROL_ROOT}/logs/infer_${experiment_name}_${STEP_TAG}.log"
    printf '[%s] inference start: experiment=%s gpu=%s\n' \
      "$(timestamp)" "${experiment_name}" "${gpu}"
    TEST_LIST="${TEST_LIST}" \
      NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
      STEP_OUTPUT_DIR_NAME="${experiment_name}_${STEP_TAG}_steps${NUM_INFERENCE_STEPS}_512x896_ctx08_49f" \
      bash "${EXPERIMENT_ROOT}/run_infer_from_experiment.sh" \
        "${checkpoint_dir}" "${gpu}" "${OUTPUT_ROOT}" \
        >"${log_path}" 2>&1
    printf '[%s] inference complete: experiment=%s log=%s\n' \
      "$(timestamp)" "${experiment_name}" "${log_path}"
  done
}

printf '[%s] watcher started: run=%s step=%s\n' \
  "$(timestamp)" "${RUN_TAG}" "${STEP_TAG}"
wait_for_checkpoints
wait_for_stable_checkpoints
selected_gpu="$(wait_for_available_gpu)"
printf '[%s] selected inference GPU: %s\n' "$(timestamp)" "${selected_gpu}"
run_inference "${selected_gpu}"
printf '[%s] all formal checkpoint inferences completed: %s\n' \
  "$(timestamp)" "${OUTPUT_ROOT}"
