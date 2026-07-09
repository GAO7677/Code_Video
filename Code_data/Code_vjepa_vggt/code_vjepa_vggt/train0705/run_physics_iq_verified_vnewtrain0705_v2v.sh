#!/usr/bin/env bash
# Physics-IQ Verified 双卡正式跑法示例:
# GPU_PAIR=0,1 \
# WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
# MODEL_NAME=train_stage1b_diffsynth_native0705_step2500_physiq_verified \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/physicsiq/train_stage1b_diffsynth_native0705 \
# VERIFIED_ROOT=/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified \
# DESCRIPTIONS_FILE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv \
# FPS=30 \
# HEIGHT=512 \
# WIDTH=896 \
# INPUT_COVER_CROP_HEIGHT=512 \
# INPUT_COVER_CROP_WIDTH=896 \
# NUM_FRAMES=150 \
# CONTEXT_FRAMES=20 \
# SAMPLING_MODE=prefix \
# NUM_INFERENCE_STEPS=40 \
# CFG_SCALE=5.0 \
# SEED=42 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v.sh
#
# 单 case smoke test:
# GPU_PAIR=0,1 \
# LIMIT=1 \
# OUTPUT_ROOT=/data/gaoya/agent-data/outputs/physicsiq_onecase_smoke \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt"
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
RUNNER_PY="${SCRIPT_DIR}/run_physics_iq_verified_vnewtrain0705_v2v.py"

GPU_PAIR="${GPU_PAIR:-${VISIBLE_GPU_IDS:-0,1}}"
INFERENCE_DEVICES="${INFERENCE_DEVICES:-cuda:0,cuda:1}"

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500}"
MODEL_NAME="${MODEL_NAME:-${METHOD_NAME:-train_stage1b_diffsynth_native0705_step2500_physiq_verified}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/test/physicsiq/train_stage1b_diffsynth_native0705}"
VERIFIED_ROOT="${VERIFIED_ROOT:-/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified}"
DESCRIPTIONS_FILE="${DESCRIPTIONS_FILE:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv}"

FPS="${FPS:-30}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-896}"
INPUT_COVER_CROP_HEIGHT="${INPUT_COVER_CROP_HEIGHT:-512}"
INPUT_COVER_CROP_WIDTH="${INPUT_COVER_CROP_WIDTH:-896}"
NUM_FRAMES="${NUM_FRAMES:-${OUTPUT_FRAMES:-150}}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-${CTX:-20}}"
SAMPLING_MODE="${SAMPLING_MODE:-prefix}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
CFG_SCALE="${CFG_SCALE:-5.0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"

RUN_NAME="${RUN_NAME:-}"
STEP_OUTPUT_DIR_NAME="${STEP_OUTPUT_DIR_NAME:-}"
METHOD_SUFFIX="${METHOD_SUFFIX:-}"
LIMIT="${LIMIT:-}"
FORCE_REPREPARE="${FORCE_REPREPARE:-0}"
KEEP_PREPARED_INPUTS="${KEEP_PREPARED_INPUTS:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
GROUNDING_DEVICE="${GROUNDING_DEVICE:-}"

check_gpu_pair_has_faulty_gpu4() {
  local raw_pair="$1"
  local clean_pair
  local gpu_id
  clean_pair="$(echo "${raw_pair}" | tr -d '[:space:]')"
  IFS=',' read -r -a pair_gpu_ids <<< "${clean_pair}"
  for gpu_id in "${pair_gpu_ids[@]}"; do
    if [ "${gpu_id}" = "4" ]; then
      echo "ERROR: gpu4 故障, 禁止使用。当前 GPU pair=${raw_pair}" >&2
      exit 1
    fi
  done
}

prepare_launch_layout() {
  local raw_pair="$1"
  local requested_inference_devices="$2"
  local -n out_visible_gpu_ids="$3"
  local -n out_inference_devices="$4"
  local clean_pair
  local gpu_id
  local -a pair_gpu_ids=()
  local -a unique_gpu_ids=()

  clean_pair="$(echo "${raw_pair}" | tr -d '[:space:]')"
  IFS=',' read -r -a pair_gpu_ids <<< "${clean_pair}"
  for gpu_id in "${pair_gpu_ids[@]}"; do
    if [ -z "${gpu_id}" ]; then
      continue
    fi
    if [ "${#unique_gpu_ids[@]}" -eq 0 ] || [ "${unique_gpu_ids[-1]}" != "${gpu_id}" ]; then
      unique_gpu_ids+=("${gpu_id}")
    fi
  done

  if [ "${#unique_gpu_ids[@]}" -eq 0 ]; then
    echo "ERROR: GPU pair 解析后为空: ${raw_pair}" >&2
    exit 1
  fi

  out_visible_gpu_ids="$(IFS=,; echo "${unique_gpu_ids[*]}")"
  if [ "${#unique_gpu_ids[@]}" -lt 2 ]; then
    out_inference_devices="none"
  else
    out_inference_devices="${requested_inference_devices}"
  fi
}

normalize_ctx_values() {
  local raw_list="$1"
  local -n out_array="$2"
  local raw_ctx
  local ctx
  out_array=()
  IFS=',' read -r -a raw_ctx_values <<< "${raw_list}"
  if [ "${#raw_ctx_values[@]}" -eq 0 ]; then
    echo "ERROR: CTX 不能为空" >&2
    exit 1
  fi
  for raw_ctx in "${raw_ctx_values[@]}"; do
    ctx="$(echo "${raw_ctx}" | xargs)"
    if [ -z "${ctx}" ]; then
      continue
    fi
    if ! [[ "${ctx}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: 非法 CTX 值: ${ctx}" >&2
      exit 1
    fi
    if [ "${ctx}" -le 0 ]; then
      echo "ERROR: CTX 必须为正整数，收到 ${ctx}" >&2
      exit 1
    fi
    out_array+=("${ctx}")
  done
  if [ "${#out_array[@]}" -eq 0 ]; then
    echo "ERROR: CTX 解析后为空" >&2
    exit 1
  fi
}

run_single_ctx() {
  local ctx_value="$1"
  local run_output_root="$2"
  local run_model_name="$3"
  local run_step_output_dir_name="$4"
  local run_name="$5"
  local launch_visible_gpu_ids=""
  local launch_inference_devices=""
  local -a cmd

  check_gpu_pair_has_faulty_gpu4 "${GPU_PAIR}"
  prepare_launch_layout "${GPU_PAIR}" "${INFERENCE_DEVICES}" launch_visible_gpu_ids launch_inference_devices

  if [ -z "${GROUNDING_DEVICE}" ] && [ -n "${launch_inference_devices}" ] && [ "${launch_inference_devices}" != "none" ]; then
    GROUNDING_DEVICE="cuda:1"
  fi

  cmd=(
    env
    PYTHONNOUSERSITE=1
    PYTHONPATH="${REPO_ROOT}:${DIFFSYNTH_ROOT}"
    CUDA_VISIBLE_DEVICES="${launch_visible_gpu_ids}"
    "${PYTHON_BIN}"
    "${RUNNER_PY}"
    --weights-root "${WEIGHTS_ROOT}"
    --model-name "${run_model_name}"
    --output-root "${run_output_root}"
    --verified-root "${VERIFIED_ROOT}"
    --descriptions-file "${DESCRIPTIONS_FILE}"
    --fps "${FPS}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --input-cover-crop-height "${INPUT_COVER_CROP_HEIGHT}"
    --input-cover-crop-width "${INPUT_COVER_CROP_WIDTH}"
    --num-frames "${NUM_FRAMES}"
    --context-frames "${ctx_value}"
    --sampling-mode "${SAMPLING_MODE}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --cfg-scale "${CFG_SCALE}"
    --negative-prompt "${NEGATIVE_PROMPT}"
    --seed "${SEED}"
    --device "${DEVICE}"
  )

  if [ -n "${launch_inference_devices}" ] && [ "${launch_inference_devices}" != "none" ]; then
    cmd+=(--inference-devices "${launch_inference_devices}")
  fi
  if [ -n "${GROUNDING_DEVICE}" ]; then
    cmd+=(--grounding-device "${GROUNDING_DEVICE}")
  fi
  if [ -n "${run_name}" ]; then
    cmd+=(--run-name "${run_name}")
  fi
  if [ -n "${run_step_output_dir_name}" ]; then
    cmd+=(--step-output-dir-name "${run_step_output_dir_name}")
  fi
  if [ -n "${METHOD_SUFFIX}" ]; then
    cmd+=(--method-suffix "${METHOD_SUFFIX}")
  fi
  if [ -n "${LIMIT}" ]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [ "${FORCE_REPREPARE}" = "1" ]; then
    cmd+=(--force-reprepare)
  fi
  if [ "${KEEP_PREPARED_INPUTS}" = "1" ]; then
    cmd+=(--keep-prepared-inputs)
  fi
  if [ "${PREPARE_ONLY}" = "1" ]; then
    cmd+=(--prepare-only)
  fi

  cmd+=("$@")

  echo "[physicsiq] gpu_pair=${GPU_PAIR}"
  echo "[physicsiq] ctx=${ctx_value}"
  echo "[physicsiq] cuda_visible_devices=${launch_visible_gpu_ids} inference_devices=${launch_inference_devices}"
  echo "[physicsiq] grounding_device=${GROUNDING_DEVICE:-<default-main>}"
  echo "[physicsiq] output_root=${run_output_root}"
  echo "[physicsiq] model_name=${run_model_name}"
  echo "[physicsiq] command: ${cmd[*]}"
  "${cmd[@]}"
}

declare -a ctx_values=()
normalize_ctx_values "${CTX}" ctx_values

if [ "${#ctx_values[@]}" -eq 1 ]; then
  single_ctx="${ctx_values[0]}"
  single_output_root="${OUTPUT_ROOT}"
  single_model_name="${MODEL_NAME}"
  single_step_output_dir_name="${STEP_OUTPUT_DIR_NAME:-step-002500}"
  single_run_name="${RUN_NAME:-train_stage1b_diffsynth_native0705_step2500_physiq_verified-bpp-run_01}"
  run_single_ctx "${single_ctx}" "${single_output_root}" "${single_model_name}" "${single_step_output_dir_name}" "${single_run_name}" "$@"
else
  for ctx_value in "${ctx_values[@]}"; do
    ctx_tag="$(printf 'ctx%02d' "${ctx_value}")"
    run_output_root="${OUTPUT_ROOT}/${ctx_tag}"
    run_model_name="${MODEL_NAME}_${ctx_tag}"
    run_step_output_dir_name="${STEP_OUTPUT_DIR_NAME:-step-002500}"
    run_name="${RUN_NAME:-train_stage1b_diffsynth_native0705_step2500_physiq_verified-${ctx_tag}-bpp-run_01}"
    run_single_ctx "${ctx_value}" "${run_output_root}" "${run_model_name}" "${run_step_output_dir_name}" "${run_name}" "$@"
  done
fi
