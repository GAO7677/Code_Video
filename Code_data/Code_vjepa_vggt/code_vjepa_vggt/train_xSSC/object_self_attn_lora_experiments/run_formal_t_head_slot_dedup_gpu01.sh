#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_formal_t_head_slot_dedup_gpu01.sh
set -euo pipefail

export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/formal_t_head_slot_dedup_merge_gpu01.json}"
RUN_TAG="${RUN_TAG:-from_scratch_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_ROOT="${LOG_ROOT:-/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_formal_logs}"
LOG_PATH="${LOG_ROOT}/t_head70_slot_dedup_merge_gpu01_${RUN_TAG}.log"

mkdir -p "${LOG_ROOT}"
echo "[formal] config=${CONFIG}"
echo "[formal] run_tag=${RUN_TAG}"
echo "[formal] log=${LOG_PATH}"

bash "${SCRIPT_DIR}/run_train_slot_dedup_from_config.sh" \
  "${CONFIG}" \
  --run-tag "${RUN_TAG}" 2>&1 | tee "${LOG_PATH}"
