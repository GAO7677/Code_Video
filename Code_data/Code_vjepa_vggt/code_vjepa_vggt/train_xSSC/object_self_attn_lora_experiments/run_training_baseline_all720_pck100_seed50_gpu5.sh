#!/usr/bin/env bash
set -euo pipefail

HERE="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${HERE}/compute_training_baseline_all720_pck100_seed50.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/training_object_query_top30_step500}"
METRICS_ROOT="${OUTPUT_ROOT}/metrics720_seed50"
CACHE_ROOT="${OUTPUT_ROOT}/metrics100/grounded_sam2_regions"
BASELINE_CONFIG="${BASELINE_CONFIG:-/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/lora_pck32_top100_t_head_no_object_gpu67_formal/pck32_top100_from_scratch_20260805T100041Z/resolved_experiment_config.json}"

mkdir -p "${METRICS_ROOT}/logs"
export CUDA_VISIBLE_DEVICES=5
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/gaoya/Code_Video/DiffTrack-main:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Grounded-SAM-2-main"

"${PYTHON}" "${WORKER}" prepare --output-root "${OUTPUT_ROOT}"
"${PYTHON}" "${WORKER}" worker \
  --output-root "${OUTPUT_ROOT}" --cache-root "${CACHE_ROOT}" \
  --baseline-config "${BASELINE_CONFIG}" --device cuda:0 \
  > "${METRICS_ROOT}/logs/gpu5.log" 2>&1
