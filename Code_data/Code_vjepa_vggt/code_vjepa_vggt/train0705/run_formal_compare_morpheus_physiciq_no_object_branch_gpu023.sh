#!/usr/bin/env bash
set -euo pipefail

# Formal batch run for the no-object-branch ablation on:
# - morpheus_real_world
# - physicIQ
#
# Methods:
# - train0705 step-002500 (disable object branch at inference)
# - train0705 step-007000 (disable object branch at inference)
#
# Execution policy:
# - Use only GPU 0, 2, 3
# - Run up to 3 inference jobs in parallel
# - After each dataset finishes, run bench.sh on that dataset root
# - Keep formal outputs grouped by dataset under one parent directory
# - Keep transient logs under /data/gaoya/agent-data

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_no_object_branch_gpu023.sh

PY=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
REPO=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
BENCH_SCRIPT="${REPO}/code_vjepa_vggt/AAAinfer/bench.sh"

LIST_MORPHEUS=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt
LIST_PHYSICIQ=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt

TRAIN0705_CHECKPOINT_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints

RESULT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare_no_object_branch
LOG_BASE=/data/gaoya/agent-data/outputs/train0705_formal_compare_no_object_branch_logs_20260706
mkdir -p "${RESULT_BASE}" "${LOG_BASE}"

GPU_POOL=(1 2)
declare -A GPU_PID=()
declare -A GPU_LABEL=()
FAILED_JOBS=()

check_freed_gpus() {
  local gpu
  for gpu in "${GPU_POOL[@]}"; do
    local pid="${GPU_PID[$gpu]:-}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then
        echo "[queue] finished label=${GPU_LABEL[$gpu]} gpu=${gpu}" >&2
      else
        local rc=$?
        echo "[queue] failed label=${GPU_LABEL[$gpu]} gpu=${gpu} rc=${rc}" >&2
        FAILED_JOBS+=("${GPU_LABEL[$gpu]}:${rc}")
      fi
      unset 'GPU_PID[$gpu]'
      unset 'GPU_LABEL[$gpu]'
    fi
  done
}

wait_for_free_gpu() {
  while :; do
    check_freed_gpus
    local gpu
    for gpu in "${GPU_POOL[@]}"; do
      if [[ -z "${GPU_PID[$gpu]:-}" ]]; then
        echo "${gpu}"
        return 0
      fi
    done
    sleep 10
  done
}

launch_job() {
  local label="$1"
  local log_path="$2"
  shift 2

  local gpu
  gpu="$(wait_for_free_gpu)"

  echo "[launch] label=${label} gpu=${gpu} log=${log_path}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "$@"
  ) >"${log_path}" 2>&1 &

  local pid=$!
  GPU_PID["${gpu}"]="${pid}"
  GPU_LABEL["${gpu}"]="${label}"
}

wait_all_jobs() {
  while ((${#GPU_PID[@]} > 0)); do
    check_freed_gpus
    if ((${#GPU_PID[@]} > 0)); then
      sleep 10
    fi
  done
}

run_train0705_step_no_object_branch() {
  local dataset_tag="$1"
  local list_path="$2"
  local step_name="$3"
  local output_root="${RESULT_BASE}/${dataset_tag}/train_stage1b_diffsynth_native0705_0705_no_object_branch"
  local log_path="${LOG_BASE}/${dataset_tag}_train0705_${step_name}_no_object_branch.log"

  mkdir -p "${output_root}"
  launch_job "${dataset_tag}:train0705_${step_name}_no_object_branch" "${log_path}" \
    env PYTHONPATH="${REPO}:${DIFFSYNTH}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py" \
      --weights-root "${TRAIN0705_CHECKPOINT_ROOT}/${step_name}" \
      --input-json-list-path "${list_path}" \
      --model-name "train_stage1b_diffsynth_native0705_0705_no_object_branch_${dataset_tag}" \
      --output-root "${output_root}" \
      --num-inference-steps 40 \
      --seed 42 \
      --disable-object-branch
}

run_dataset() {
  local dataset_tag="$1"
  local list_path="$2"
  local dataset_root="${RESULT_BASE}/${dataset_tag}"

  echo "============================================================"
  echo "[dataset:start] ${dataset_tag}"
  echo "[dataset:list]  ${list_path}"
  echo "[dataset:root]  ${dataset_root}"
  echo "============================================================"

  run_train0705_step_no_object_branch "${dataset_tag}" "${list_path}" "step-002500"
  run_train0705_step_no_object_branch "${dataset_tag}" "${list_path}" "step-007000"

  wait_all_jobs

  if ((${#FAILED_JOBS[@]} > 0)); then
    echo "[dataset:warning] some jobs failed before bench: ${FAILED_JOBS[*]}" >&2
  fi

  echo "[bench:start] dataset=${dataset_tag}"
  CUDA_VISIBLE_DEVICES=0 BENCH_CUDA_VISIBLE_DEVICES=0 bash "${BENCH_SCRIPT}" "${dataset_root}"
  echo "[bench:done] dataset=${dataset_tag}"
}

run_dataset "morpheus_real_world" "${LIST_MORPHEUS}"
run_dataset "physicIQ" "${LIST_PHYSICIQ}"

if ((${#FAILED_JOBS[@]} > 0)); then
  echo "[all_done_with_errors] ${FAILED_JOBS[*]}" >&2
  exit 1
fi

echo "[all_done] no-object-branch formal comparison finished successfully"
