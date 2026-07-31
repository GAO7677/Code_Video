#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_finish_step3000_eval_gpu01235.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/xssc_lora_three_train_watch_config.json}"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
GPUS="${GPUS:-0,1,2,3,5}"
METHODS="${METHODS:-full_sa_resume,s_head59_resume}"
STEPS="${STEPS:-3000}"

cd "${SCRIPT_DIR}"
echo "[finish-eval] config=${CONFIG}"
echo "[finish-eval] gpus=${GPUS} methods=${METHODS} steps=${STEPS}"

"${PYTHON}" "${SCRIPT_DIR}/xssc_lora_physiciq_sharded_catchup.py" \
  --config "${CONFIG}" \
  --gpus "${GPUS}" \
  --methods "${METHODS}" \
  --steps "${STEPS}"

"${PYTHON}" "${SCRIPT_DIR}/xssc_lora_checkpoint_parallel_metrics.py" \
  --config "${CONFIG}" \
  --gpus "${GPUS}" \
  --methods "${METHODS}" \
  --steps "${STEPS}" \
  --workers-per-gpu 1 \
  --refresh

"${PYTHON}" "${SCRIPT_DIR}/xssc_lora_physiciq_parallel_metrics.py" \
  --config "${CONFIG}" \
  --kind cpu \
  --methods "${METHODS}" \
  --steps "${STEPS}" \
  --cpu-workers 8 \
  --skip-locked \
  --refresh-plots

"${PYTHON}" "${SCRIPT_DIR}/xssc_lora_physiciq_parallel_metrics.py" \
  --config "${CONFIG}" \
  --kind gpu \
  --gpus "${GPUS}" \
  --methods "${METHODS}" \
  --steps "${STEPS}" \
  --workers-per-gpu 1 \
  --skip-locked \
  --refresh-plots

"${PYTHON}" "${SCRIPT_DIR}/build_xssc_lora_checkpoint_dashboard.py" \
  --config "${CONFIG}"

echo "[finish-eval] done"
