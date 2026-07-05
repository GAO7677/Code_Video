#!/usr/bin/env bash
set -euo pipefail

# Formal batch run for:
# - morpheus_real_world (123 cases)
# - physicIQ (67 cases)
# Methods:
# - baseline: wan2p2_ti2v5B
# - openvid_lora_10000
# - 0613pybullet_lora_000500
# - train0705 step-002500
# - train0705 step-007000
#
# Execution policy:
# - Use only GPU 0, 2, 3
# - Run up to 3 inference jobs in parallel
# - After each dataset finishes, run bench.sh on that dataset root
# - Keep formal outputs grouped by dataset under one parent directory
# - Keep transient logs under /data/gaoya/agent-data

PY=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
REPO=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
WAN22=/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main
TRAIN0419=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
BENCH_SCRIPT="${REPO}/code_vjepa_vggt/AAAinfer/bench.sh"

LIST_MORPHEUS=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt
LIST_PHYSICIQ=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt

BASE_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500
OPENVID_LORA_ROOT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000
TRAIN0705_CHECKPOINT_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints

RESULT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare
RUNTIME_BASE=/data/gaoya/agent-data/outputs/train0705_formal_compare_runtime_20260705
LOG_BASE=/data/gaoya/agent-data/outputs/train0705_formal_compare_logs_20260705
mkdir -p "${RESULT_BASE}" "${RUNTIME_BASE}" "${LOG_BASE}"

GPU_POOL=(0 2 3)
declare -A GPU_PID=()
declare -A GPU_LABEL=()
declare -A PID_GPU=()
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
      unset 'PID_GPU[$pid]'
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
  PID_GPU["${pid}"]="${gpu}"
}

wait_all_jobs() {
  while ((${#GPU_PID[@]} > 0)); do
    check_freed_gpus
    if ((${#GPU_PID[@]} > 0)); then
      sleep 10
    fi
  done
}

run_baseline() {
  local dataset_tag="$1"
  local list_path="$2"
  local output_root="${RESULT_BASE}/${dataset_tag}/basemodel/wan2p2_ti2v5B"
  local log_path="${LOG_BASE}/${dataset_tag}_wan2p2_ti2v5B.log"

  mkdir -p "${output_root}"
  launch_job "${dataset_tag}:wan2p2_ti2v5B" "${log_path}" \
    env PYTHONPATH="${REPO}:${WAN22}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/AAAinfer/wanti2v.py" \
      --input-list "${list_path}" \
      --model-name "wan2p2_ti2v5B_${dataset_tag}" \
      --output-root "${output_root}" \
      --frame-num 25 \
      --sampling-steps 40 \
      --cfg-scale 5.0 \
      --fps 30 \
      --seed 42 \
      --offload-model
}

run_openvid() {
  local dataset_tag="$1"
  local list_path="$2"
  local output_root="${RESULT_BASE}/${dataset_tag}/loramodel/wan_openvid_lorav2v_step10000"
  local runtime_root="${RUNTIME_BASE}/${dataset_tag}/loramodel/wan_openvid_lorav2v_step10000_runtime"
  local log_path="${LOG_BASE}/${dataset_tag}_wan_openvid_lorav2v_step10000.log"

  mkdir -p "${output_root}" "${runtime_root}"
  launch_job "${dataset_tag}:wan_openvid_lorav2v_step10000" "${log_path}" \
    env PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_lorav2v.py" \
      --weights-root "${OPENVID_LORA_ROOT}" \
      --input-json-list-path "${list_path}" \
      --model-name "wan_openvid_lorav2v_step10000_${dataset_tag}" \
      --output-root "${output_root}" \
      --runtime-root "${runtime_root}" \
      --num-frames 25 \
      --num-inference-steps 40 \
      --cfg-scale 5.0 \
      --seed 42
}

run_pybullet_lora() {
  local dataset_tag="$1"
  local list_path="$2"
  local output_root="${RESULT_BASE}/${dataset_tag}/loramodel/wan_openvid_0613pybullet_lorav2v_step000500"
  local runtime_root="${RUNTIME_BASE}/${dataset_tag}/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_runtime"
  local log_path="${LOG_BASE}/${dataset_tag}_wan_openvid_0613pybullet_lorav2v_step000500.log"

  mkdir -p "${output_root}" "${runtime_root}"
  launch_job "${dataset_tag}:wan_openvid_0613pybullet_lorav2v_step000500" "${log_path}" \
    env PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_0613pybullet_lorav2v.py" \
      --weights-root "${BASE_LORA_ROOT}" \
      --input-json-list-path "${list_path}" \
      --model-name "wan_openvid_0613pybullet_lorav2v_step000500_${dataset_tag}" \
      --output-root "${output_root}" \
      --runtime-root "${runtime_root}" \
      --num-frames 25 \
      --num-inference-steps 40 \
      --cfg-scale 5.0 \
      --seed 42
}

run_train0705_step() {
  local dataset_tag="$1"
  local list_path="$2"
  local step_name="$3"
  local output_root="${RESULT_BASE}/${dataset_tag}/train_stage1b_diffsynth_native0705_0705"
  local log_path="${LOG_BASE}/${dataset_tag}_train0705_${step_name}.log"

  mkdir -p "${output_root}"
  launch_job "${dataset_tag}:train0705_${step_name}" "${log_path}" \
    env PYTHONPATH="${REPO}:${DIFFSYNTH}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py" \
      --weights-root "${TRAIN0705_CHECKPOINT_ROOT}/${step_name}" \
      --input-json-list-path "${list_path}" \
      --model-name "train_stage1b_diffsynth_native0705_0705_${dataset_tag}" \
      --output-root "${output_root}" \
      --num-inference-steps 40 \
      --seed 42
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

  run_baseline "${dataset_tag}" "${list_path}"
  run_openvid "${dataset_tag}" "${list_path}"
  run_pybullet_lora "${dataset_tag}" "${list_path}"
  run_train0705_step "${dataset_tag}" "${list_path}" "step-002500"
  run_train0705_step "${dataset_tag}" "${list_path}" "step-007000"

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

echo "[all_done] formal comparison finished successfully"
