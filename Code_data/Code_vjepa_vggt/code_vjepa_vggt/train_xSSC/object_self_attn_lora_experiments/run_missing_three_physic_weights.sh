#!/usr/bin/env bash
# Run selected missing PhysicIQ tasks for the three-method watcher experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${CONFIG:-${SCRIPT_DIR}/xssc_lora_three_train_watch_config_with_t_head.json}"
GPUS="${GPUS:-5,7}"
METHODS="${METHODS:-slot_dedup_merge,t_head70}"
STEPS="${STEPS:-500,1000,1500,3000,3500}"
CPU_WORKERS="${CPU_WORKERS:-8}"

if [[ ! -s "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi

echo "[missing-physic] config=${CONFIG}"
echo "[missing-physic] gpus=${GPUS} methods=${METHODS} steps=${STEPS}"

"${PYTHON}" "${SCRIPT_DIR}/xssc_lora_physiciq_sharded_catchup.py" \
  --config "${CONFIG}" \
  --gpus "${GPUS}" \
  --methods "${METHODS}" \
  --steps "${STEPS}" \
  --force-missing

"${PYTHON}" "${SCRIPT_DIR}/xssc_lora_physiciq_parallel_metrics.py" \
  --config "${CONFIG}" \
  --kind cpu \
  --methods "${METHODS}" \
  --steps "${STEPS}" \
  --cpu-workers "${CPU_WORKERS}" \
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

echo "[missing-physic] done"
