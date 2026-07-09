#!/usr/bin/env bash
set -euo pipefail

# Unified-parameter batch inference for 3 methods on 2 datasets:
# - baseline: wan2p2_ti2v5B
# - lora: wan_openvid_lorav2v_step10000
# - lora: wan_openvid_0613pybullet_lorav2v_step000500
#
# Unified generation params:
# - height/width: 512x896
# - num_frames: 49
# - num_inference_steps: 40
# - cfg_scale: 5.0
# - fps: 30
# - seed: 42
# - negative_prompt: Chinese default prompt below
#
# Example:
# GPU_POOL="0 2 3" \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_wan_base_two_loras_aligned49.sh

PY=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
REPO=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
WAN22=/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main
TRAIN0419=/home/gaoya/Code_Video/Code_data/Code_train/train_0419

LIST_MORPHEUS=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt
LIST_PHYSICIQ=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt

BASE_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500
OPENVID_LORA_ROOT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000

RESULT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare_aligned49
RUNTIME_BASE=/data/gaoya/agent-data/outputs/train0705_formal_compare_aligned49_runtime
LOG_BASE=/data/gaoya/agent-data/outputs/train0705_formal_compare_aligned49_logs
mkdir -p "${RESULT_BASE}" "${RUNTIME_BASE}" "${LOG_BASE}"

GPU_POOL_STR="${GPU_POOL:-0 2 3}"
read -r -a GPU_POOL <<< "${GPU_POOL_STR}"

NEGATIVE_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

declare -A GPU_PID=()
declare -A GPU_LABEL=()

check_freed_gpus() {
  local gpu
  for gpu in "${GPU_POOL[@]}"; do
    local pid="${GPU_PID[$gpu]:-}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}"
      echo "[queue] finished label=${GPU_LABEL[$gpu]} gpu=${gpu}" >&2
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
  GPU_PID["${gpu}"]=$!
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

run_baseline() {
  local dataset_tag="$1"
  local list_path="$2"
  local output_root="${RESULT_BASE}/${dataset_tag}/basemodel/wan2p2_ti2v5B_aligned49"
  local log_path="${LOG_BASE}/${dataset_tag}_wan2p2_ti2v5B_aligned49.log"

  mkdir -p "${output_root}"
  launch_job "${dataset_tag}:wan2p2_ti2v5B_aligned49" "${log_path}" \
    env PYTHONPATH="${REPO}:${WAN22}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/AAAinfer/wanti2v.py" \
      --input-list "${list_path}" \
      --model-name "wan2p2_ti2v5B_${dataset_tag}_aligned49" \
      --output-root "${output_root}" \
      --size 512*896 \
      --frame-num 49 \
      --sampling-steps 40 \
      --cfg-scale 5.0 \
      --fps 30 \
      --seed 42 \
      --negative-prompt "${NEGATIVE_PROMPT}" \
      --offload-model
}

run_openvid() {
  local dataset_tag="$1"
  local list_path="$2"
  local output_root="${RESULT_BASE}/${dataset_tag}/loramodel/wan_openvid_lorav2v_step10000_aligned49"
  local runtime_root="${RUNTIME_BASE}/${dataset_tag}/loramodel/wan_openvid_lorav2v_step10000_aligned49_runtime"
  local log_path="${LOG_BASE}/${dataset_tag}_wan_openvid_lorav2v_step10000_aligned49.log"

  mkdir -p "${output_root}" "${runtime_root}"
  launch_job "${dataset_tag}:wan_openvid_lorav2v_step10000_aligned49" "${log_path}" \
    env PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_lorav2v.py" \
      --weights-root "${OPENVID_LORA_ROOT}" \
      --input-json-list-path "${list_path}" \
      --model-name "wan_openvid_lorav2v_step10000_${dataset_tag}_aligned49" \
      --output-root "${output_root}" \
      --runtime-root "${runtime_root}" \
      --height 512 \
      --width 896 \
      --num-frames 49 \
      --num-inference-steps 40 \
      --cfg-scale 5.0 \
      --fps 30 \
      --seed 42 \
      --negative-prompt "${NEGATIVE_PROMPT}"
}

run_pybullet_lora() {
  local dataset_tag="$1"
  local list_path="$2"
  local output_root="${RESULT_BASE}/${dataset_tag}/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_aligned49"
  local runtime_root="${RUNTIME_BASE}/${dataset_tag}/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_aligned49_runtime"
  local log_path="${LOG_BASE}/${dataset_tag}_wan_openvid_0613pybullet_lorav2v_step000500_aligned49.log"

  mkdir -p "${output_root}" "${runtime_root}"
  launch_job "${dataset_tag}:wan_openvid_0613pybullet_lorav2v_step000500_aligned49" "${log_path}" \
    env PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
    "${PY}" \
    "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_0613pybullet_lorav2v.py" \
      --weights-root "${BASE_LORA_ROOT}" \
      --input-json-list-path "${list_path}" \
      --model-name "wan_openvid_0613pybullet_lorav2v_step000500_${dataset_tag}_aligned49" \
      --output-root "${output_root}" \
      --runtime-root "${runtime_root}" \
      --height 512 \
      --width 896 \
      --num-frames 49 \
      --num-inference-steps 40 \
      --cfg-scale 5.0 \
      --fps 30 \
      --seed 42 \
      --negative-prompt "${NEGATIVE_PROMPT}"
}

run_dataset() {
  local dataset_tag="$1"
  local list_path="$2"

  echo "============================================================"
  echo "[dataset:start] ${dataset_tag}"
  echo "[dataset:list]  ${list_path}"
  echo "============================================================"

  run_baseline "${dataset_tag}" "${list_path}"
  run_openvid "${dataset_tag}" "${list_path}"
  run_pybullet_lora "${dataset_tag}" "${list_path}"
  wait_all_jobs

  echo "[dataset:done] ${dataset_tag}"
}

run_dataset "morpheus_real_world" "${LIST_MORPHEUS}"
run_dataset "physicIQ" "${LIST_PHYSICIQ}"

echo "[all_done] aligned49 inference finished successfully"
