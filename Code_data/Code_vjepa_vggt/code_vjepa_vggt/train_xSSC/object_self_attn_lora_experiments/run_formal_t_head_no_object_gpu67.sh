#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_formal_t_head_no_object_gpu67.sh

set -euo pipefail

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments"
PYTHON_BIN="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
RUN_TAG="${RUN_TAG:-from_scratch_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_ROOT="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_formal_logs"
LOG_PATH="${LOG_ROOT}/t_head_no_object_gpu67_${RUN_TAG}.log"

mkdir -p "${LOG_ROOT}"
export PYTHONNOUSERSITE=1

"${PYTHON_BIN}" "${ROOT}/launch_from_config.py" \
  "${ROOT}/configs/formal_t_head_no_object_gpu67.json" \
  --run-tag "${RUN_TAG}" 2>&1 | tee "${LOG_PATH}"
