#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-wan}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/Benchmark/physics_IQ}"
WAN_ROOT="${WAN_ROOT:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B}"

TI2V_MODEL_NAME="${TI2V_MODEL_NAME:-wan22_ti2v_5b_physicsiq}"
TV2V_MODEL_NAME="${TV2V_MODEL_NAME:-wan22_tv2v_pretrain_ctx8_physicsiq}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/physics_iq_method_summary.csv}"
NUM_FRAMES="${NUM_FRAMES:-161}"

TI2V_PY="/home/gaoya/Code_Video/Code_data/Code_benchmark/physicalIQ/wan22_ti2v_physics_iq_eval_multigpu.py"
TV2V_PY="/home/gaoya/Code_Video/Code_data/Code_benchmark/physicalIQ/wan22_tv2v_physics_iq_eval_multigpu.py"
SUMMARY_PY="/home/gaoya/Code_Video/Code_data/Code_benchmark/physicalIQ/export_physics_iq_method_summary.py"

mkdir -p "${OUTPUT_ROOT}"

echo "[physics_iq] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[physics_iq] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[physics_iq] TI2V_MODEL_NAME=${TI2V_MODEL_NAME}"
echo "[physics_iq] TV2V_MODEL_NAME=${TV2V_MODEL_NAME}"
echo "[physics_iq] NUM_FRAMES=${NUM_FRAMES}"

export CUDA_VISIBLE_DEVICES

conda run -n "${CONDA_ENV}" python "${TI2V_PY}" \
  --multi_gpu \
  --model_root "${WAN_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --height 720 \
  --width 1280 \
  --fps 30 \
  --num_frames "${NUM_FRAMES}" \
  --seed 42 \
  --model_name "${TI2V_MODEL_NAME}"

conda run -n "${CONDA_ENV}" python "${TV2V_PY}" \
  --multi_gpu \
  --wan_root "${WAN_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --height 720 \
  --width 1280 \
  --fps 30 \
  --num_frames "${NUM_FRAMES}" \
  --context_frames 8 \
  --seed 42 \
  --model_name "${TV2V_MODEL_NAME}"

conda run -n "${CONDA_ENV}" python "${SUMMARY_PY}" \
  --output_root "${OUTPUT_ROOT}" \
  --summary_csv "${SUMMARY_CSV}" \
  --method "${TI2V_MODEL_NAME}:TI2V:" \
  --method "${TV2V_MODEL_NAME}:TV2V:8"

echo "[physics_iq] summary csv: ${SUMMARY_CSV}"



