#!/usr/bin/env bash
set -euo pipefail

# Run only one case per dataset per method.
# Total jobs:
# - 2 datasets
# - 5 methods
# = 10 one-case jobs
#
# Scheduling policy:
# - use only gpu0, gpu2, gpu3
# - run up to 3 jobs in parallel
# - remaining jobs wait in queue automatically

PY=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
REPO=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
WAN22=/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main
TRAIN0419=/home/gaoya/Code_Video/Code_data/Code_train/train_0419

LIST_MORPHEUS=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt
LIST_PHYSICIQ=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt

TRAIN0705_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints
OPENVID_LORA_ROOT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000
PYBULLET_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500

LOG_DIR=/data/gaoya/agent-data/outputs/train0705_onecase_queue_logs_20260705
mkdir -p "${LOG_DIR}"

ONECASE_DIR=/data/gaoya/agent-data/outputs/train0705_onecase_queue_lists_20260705
mkdir -p "${ONECASE_DIR}"

OUTPUT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_onecase_compare
mkdir -p "${OUTPUT_BASE}"

TMP_RUNTIME_BASE=/data/gaoya/agent-data/outputs/train0705_onecase_compare_runtime
mkdir -p "${TMP_RUNTIME_BASE}"

PIDS=()
PID_LABELS=()

wait_for_slot() {
  while ((${#PIDS[@]} >= 3)); do
    local new_pids=()
    local new_labels=()
    local i
    for i in "${!PIDS[@]}"; do
      local pid="${PIDS[$i]}"
      local label="${PID_LABELS[$i]}"
      if kill -0 "${pid}" 2>/dev/null; then
        new_pids+=("${pid}")
        new_labels+=("${label}")
      else
        echo "[queue] finished ${label}"
      fi
    done
    PIDS=("${new_pids[@]}")
    PID_LABELS=("${new_labels[@]}")
    if ((${#PIDS[@]} >= 3)); then
      sleep 5
    fi
  done
}

launch_job() {
  local gpu="$1"
  local label="$2"
  local log_path="$3"
  shift 3

  echo "[queue] launch gpu=${gpu} label=${label}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "$@"
  ) >"${log_path}" 2>&1 &
  local pid=$!
  PIDS+=("${pid}")
  PID_LABELS+=("${label}")
  echo "[queue] pid=${pid} log=${log_path}"
}

schedule_dataset_jobs() {
  local dataset_tag="$1"
  local list_path="$2"
  local onecase_list="${ONECASE_DIR}/${dataset_tag}_onecase.txt"
  local dataset_output_root="${OUTPUT_BASE}/${dataset_tag}"
  local dataset_runtime_root="${TMP_RUNTIME_BASE}/${dataset_tag}"
  head -n 1 "${list_path}" > "${onecase_list}"

  wait_for_slot
  launch_job 0 "${dataset_tag}_baseline" "${LOG_DIR}/${dataset_tag}_baseline.log" \
    env PYTHONPATH="${REPO}:${WAN22}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/AAAinfer/wanti2v.py" \
      --input-list "${onecase_list}" \
      --model-name "wan2p2_ti2v5B_${dataset_tag}" \
      --output-root "${dataset_output_root}/basemodel/wan2p2_ti2v5B" \
      --frame-num 25 \
      --sampling-steps 40 \
      --cfg-scale 5.0 \
      --fps 30 \
      --seed 42 \
      --offload-model

  wait_for_slot
  launch_job 2 "${dataset_tag}_openvid_lora_10000" "${LOG_DIR}/${dataset_tag}_openvid_lora_10000.log" \
    env PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_lorav2v.py" \
      --weights-root "${OPENVID_LORA_ROOT}" \
      --input-json-list-path "${onecase_list}" \
      --model-name "wan_openvid_lorav2v_step10000_${dataset_tag}" \
      --output-root "${dataset_output_root}/loramodel/wan_openvid_lorav2v_step10000" \
      --runtime-root "${dataset_runtime_root}/loramodel/wan_openvid_lorav2v_step10000_runtime" \
      --num-frames 25 \
      --num-inference-steps 40 \
      --cfg-scale 5.0 \
      --seed 42 \
      --limit 1

  wait_for_slot
  launch_job 3 "${dataset_tag}_0613pybullet_lora_000500" "${LOG_DIR}/${dataset_tag}_0613pybullet_lora_000500.log" \
    env PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_0613pybullet_lorav2v.py" \
      --weights-root "${PYBULLET_LORA_ROOT}" \
      --input-json-list-path "${onecase_list}" \
      --model-name "wan_openvid_0613pybullet_lorav2v_step000500_${dataset_tag}" \
      --output-root "${dataset_output_root}/loramodel/wan_openvid_0613pybullet_lorav2v_step000500" \
      --runtime-root "${dataset_runtime_root}/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_runtime" \
      --num-frames 25 \
      --num-inference-steps 40 \
      --cfg-scale 5.0 \
      --seed 42 \
      --limit 1

  wait_for_slot
  launch_job 0 "${dataset_tag}_train0705_step002500" "${LOG_DIR}/${dataset_tag}_train0705_step002500.log" \
    env PYTHONPATH="${REPO}:${DIFFSYNTH}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py" \
      --weights-root "${TRAIN0705_ROOT}/step-002500" \
      --input-json-list-path "${onecase_list}" \
      --model-name "train_stage1b_diffsynth_native0705_0705_${dataset_tag}" \
      --output-root "${dataset_output_root}/train_stage1b_diffsynth_native0705_0705" \
      --num-inference-steps 40 \
      --seed 42 \
      --limit 1

  wait_for_slot
  launch_job 2 "${dataset_tag}_train0705_step007000" "${LOG_DIR}/${dataset_tag}_train0705_step007000.log" \
    env PYTHONPATH="${REPO}:${DIFFSYNTH}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py" \
      --weights-root "${TRAIN0705_ROOT}/step-007000" \
      --input-json-list-path "${onecase_list}" \
      --model-name "train_stage1b_diffsynth_native0705_0705_${dataset_tag}" \
      --output-root "${dataset_output_root}/train_stage1b_diffsynth_native0705_0705" \
      --num-inference-steps 40 \
      --seed 42 \
      --limit 1
}

schedule_dataset_jobs "morpheus_real_world" "${LIST_MORPHEUS}"
schedule_dataset_jobs "physicIQ" "${LIST_PHYSICIQ}"

while ((${#PIDS[@]} > 0)); do
  new_pids=()
  new_labels=()
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${PID_LABELS[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      new_pids+=("${pid}")
      new_labels+=("${label}")
    else
      echo "[queue] finished ${label}"
    fi
  done
  PIDS=("${new_pids[@]}")
  PID_LABELS=("${new_labels[@]}")
  if ((${#PIDS[@]} > 0)); then
    sleep 5
  fi
done

echo "[all_done] logs=${LOG_DIR}"
