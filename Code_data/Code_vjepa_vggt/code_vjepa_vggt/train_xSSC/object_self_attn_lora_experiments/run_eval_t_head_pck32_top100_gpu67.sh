#!/usr/bin/env bash
# Run: bash run_eval_t_head_pck32_top100_gpu67.sh

set -euo pipefail

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
CONFIG="${ROOT}/xssc_lora_three_train_watch_config_with_t_head.json"
METHOD="t_head100_lora_pck32_no_object"
STEPS="500,1000,1500"
LOG_ROOT="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/logs/manual_t_head100_gpu67"
mkdir -p "${LOG_ROOT}"

echo "[1/5] Fill missing test_5 generation on GPU6."
"${PYTHON}" "${ROOT}/run_missing_checkpoint_generation.py" \
  --config "${CONFIG}" \
  --gpus 6 \
  --methods "${METHOD}" \
  --steps "${STEPS}" \
  --test5-only \
  2>&1 | tee "${LOG_ROOT}/01_test5_generation.log"

echo "[2/5] Fill missing PhysicIQ generation on GPU6/7."
"${PYTHON}" "${ROOT}/xssc_lora_physiciq_sharded_catchup.py" \
  --config "${CONFIG}" \
  --gpus 6,7 \
  --methods "${METHOD}" \
  --steps "${STEPS}" \
  2>&1 | tee "${LOG_ROOT}/02_physiciq_generation.log"

echo "[3/5] Run CPU metrics for both test sets."
"${PYTHON}" "${ROOT}/xssc_lora_checkpoint_filtered_cpu_metrics.py" \
  --config "${CONFIG}" \
  --methods "${METHOD}" \
  --steps "${STEPS}" \
  --workers 4 \
  --refresh \
  2>&1 | tee "${LOG_ROOT}/03_test5_cpu_metrics.log" &
test5_cpu_pid=$!
"${PYTHON}" "${ROOT}/xssc_lora_physiciq_parallel_metrics.py" \
  --config "${CONFIG}" \
  --kind cpu \
  --cpu-workers 4 \
  --methods "${METHOD}" \
  --steps "${STEPS}" \
  --skip-locked \
  --refresh-plots \
  2>&1 | tee "${LOG_ROOT}/03_physiciq_cpu_metrics.log" &
phys_cpu_pid=$!

echo "[4/5] Run missing GPU metrics on GPU6/7."
"${PYTHON}" "${ROOT}/xssc_lora_checkpoint_parallel_metrics.py" \
  --config "${CONFIG}" \
  --gpus 6,7 \
  --methods "${METHOD}" \
  --steps "${STEPS}" \
  --workers-per-gpu 2 \
  --refresh \
  2>&1 | tee "${LOG_ROOT}/04_test5_gpu_metrics.log"
"${PYTHON}" "${ROOT}/xssc_lora_physiciq_parallel_metrics.py" \
  --config "${CONFIG}" \
  --kind gpu \
  --gpus 6,7 \
  --workers-per-gpu 2 \
  --methods "${METHOD}" \
  --steps "${STEPS}" \
  --skip-locked \
  --refresh-plots \
  2>&1 | tee "${LOG_ROOT}/04_physiciq_gpu_metrics.log"

wait "${test5_cpu_pid}"
wait "${phys_cpu_pid}"

echo "[5/5] Refresh the combined dashboard."
"${PYTHON}" "${ROOT}/build_xssc_lora_checkpoint_dashboard.py" \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_ROOT}/05_refresh_dashboard.log"
echo "Evaluation complete for ${METHOD}, steps ${STEPS}."
