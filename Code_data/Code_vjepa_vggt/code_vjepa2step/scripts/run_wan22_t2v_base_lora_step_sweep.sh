#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_wan22_t2v_base_lora_step_sweep.sh
#
# Optional overrides:
#   GPU_ID=1
#   PROMPT="..."
#   BASE_SEED=20250622
#   STEPS_LIST="5 15 25 50"
#   OUTPUT_ROOT=/some/path

GPU_ID="${GPU_ID:-1}"
PROMPT="${PROMPT:-A ball flew in from the left, knocking a wooden block that was stationary on the ground far away.}"
BASE_SEED="${BASE_SEED:-20250622}"
STEPS_LIST="${STEPS_LIST:-5 15 25 50}"

ROOT_DIR="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa2step"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/tmp/wan22_t2v_base_lora_step_sweep}"

WAN22_REPO="${WAN22_REPO:-/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main}"
WAN22_CKPT="${WAN22_CKPT:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B}"

PYTHON_BIN="${PYTHON_BIN:-/data/gaoya/miniconda3/envs/wan/bin/python}"
BASE_PYTHONPATH="${BASE_PYTHONPATH:-/home/gaoya/Code_Video/Code_data/Code_train/train_0419:/home/gaoya/Code_Video/DiffSynth-Studio-main}"
T2V_LORA_SCRIPT="${T2V_LORA_SCRIPT:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/infer_t2v_lora.py}"

HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-121}"
FPS="${FPS:-24}"
CFG_SCALE="${CFG_SCALE:-5.0}"

mkdir -p "${OUTPUT_ROOT}"

declare -A LORA_PATHS=(
  ["step000500"]="/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
  ["step001000"]="/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-001000/checkpoint.safetensors"
)

assert_file() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    echo "[error] file not found: ${path}" >&2
    exit 1
  fi
}

assert_dir() {
  local path="$1"
  if [ ! -d "${path}" ]; then
    echo "[error] directory not found: ${path}" >&2
    exit 1
  fi
}

log_run_header() {
  local model_name="$1"
  local steps="$2"
  local output_path="$3"
  echo "[run] model=${model_name} steps=${steps} seed=${BASE_SEED}"
  echo "[run] output=${output_path}"
}

run_base_t2v() {
  local steps="$1"
  local run_name="wan22_base_t2v_steps${steps}_seed${BASE_SEED}"
  local output_path="${OUTPUT_ROOT}/${run_name}.mp4"
  local log_path="${OUTPUT_ROOT}/${run_name}.log"

  log_run_header "wan22_base_t2v" "${steps}" "${output_path}"

  (
    cd "${WAN22_REPO}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      "${PYTHON_BIN}" generate.py \
        --task ti2v-5B \
        --size "${WIDTH}*${HEIGHT}" \
        --ckpt_dir "${WAN22_CKPT}" \
        --offload_model True \
        --convert_model_dtype \
        --t5_cpu \
        --sample_steps "${steps}" \
        --sample_guide_scale "${CFG_SCALE}" \
        --base_seed "${BASE_SEED}" \
        --prompt "${PROMPT}" \
        --save_file "${output_path}"
  ) 2>&1 | tee "${log_path}"
}

run_lora_t2v() {
  local lora_tag="$1"
  local lora_path="$2"
  local steps="$3"
  local run_name="wan22_lora_${lora_tag}_t2v_steps${steps}_seed${BASE_SEED}"
  local output_path="${OUTPUT_ROOT}/${run_name}.mp4"
  local log_path="${OUTPUT_ROOT}/${run_name}.log"

  log_run_header "wan22_lora_${lora_tag}" "${steps}" "${output_path}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${BASE_PYTHONPATH}" \
    "${PYTHON_BIN}" "${T2V_LORA_SCRIPT}" \
      --wan_root "${WAN22_CKPT}" \
      --lora_path "${lora_path}" \
      --output_video_path "${output_path}" \
      --prompt "${PROMPT}" \
      --seed "${BASE_SEED}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num_frames "${NUM_FRAMES}" \
      --fps "${FPS}" \
      --num_inference_steps "${steps}" \
      --cfg_scale "${CFG_SCALE}" \
      --overwrite \
      2>&1 | tee "${log_path}"
}

main() {
  assert_dir "${WAN22_REPO}"
  assert_dir "${WAN22_CKPT}"
  assert_file "${PYTHON_BIN}"
  assert_file "${T2V_LORA_SCRIPT}"
  for lora_path in "${LORA_PATHS[@]}"; do
    assert_file "${lora_path}"
  done

  echo "[info] gpu_id=${GPU_ID}"
  echo "[info] prompt=${PROMPT}"
  echo "[info] seed=${BASE_SEED}"
  echo "[info] steps=${STEPS_LIST}"
  echo "[info] output_root=${OUTPUT_ROOT}"

  for steps in ${STEPS_LIST}; do
    run_base_t2v "${steps}"
    for lora_tag in $(printf '%s\n' "${!LORA_PATHS[@]}" | sort); do
      run_lora_t2v "${lora_tag}" "${LORA_PATHS[${lora_tag}]}" "${steps}"
    done
  done

  echo "[done] completed wan2.2 base+lora t2v step sweep"
}

main "$@"
