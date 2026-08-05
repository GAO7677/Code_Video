#!/usr/bin/env bash
# Capture Top30 object-query attention from the training baseline on GPU5.
set -euo pipefail

HERE="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${HERE}/capture_training_object_query_top30_step500.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/training_object_query_top30_step500}"
CACHE_ROOT="${OUTPUT_ROOT}/grounded_sam2_regions"
BASELINE_CONFIG="${BASELINE_CONFIG:-/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/lora_pck32_top100_t_head_no_object_gpu67_formal/pck32_top100_from_scratch_20260805T100041Z/resolved_experiment_config.json}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-8000}"

mkdir -p "${OUTPUT_ROOT}/logs"
export CUDA_VISIBLE_DEVICES=5
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/gaoya/Code_Video/DiffTrack-main:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Grounded-SAM-2-main"

status() {
  local state="$1" message="$2"
  printf '{"state":"%s","message":"%s"}\n' "${state}" "${message}" > "${OUTPUT_ROOT}/status.json.tmp"
  mv "${OUTPUT_ROOT}/status.json.tmp" "${OUTPUT_ROOT}/status.json"
}

wait_gpu5() {
  local used
  while true; do
    used="$(nvidia-smi -i 5 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${used}" -le "${GPU_MEMORY_LIMIT_MIB}" ]]; then
      return
    fi
    status waiting_gpu "GPU5 using ${used} MiB; waiting below ${GPU_MEMORY_LIMIT_MIB} MiB"
    sleep "${POLL_SECONDS}"
  done
}

"${PYTHON}" "${WORKER}" prepare --output-root "${OUTPUT_ROOT}"

if [[ ! -s "${CACHE_ROOT}/complete.marker" ]]; then
  wait_gpu5
  status caching "GPU5: GroundingDINO and SAM2 object cache"
  "${PYTHON}" /home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/precompute_toydataset_sam2_regions.py \
    --dataset-root "${OUTPUT_ROOT}/dataset" \
    --cache-root "${CACHE_ROOT}" \
    --worker-id 0 --num-workers 1 --device cuda:0 \
    > "${OUTPUT_ROOT}/logs/sam2_cache.log" 2>&1
  date -u +%FT%TZ > "${CACHE_ROOT}/complete.marker"
fi

wait_gpu5
status loading_baseline "GPU5: loading Wan2.2 + OpenVid LoRA baseline without Top100 modules"
"${PYTHON}" "${WORKER}" capture \
  --output-root "${OUTPUT_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --baseline-config "${BASELINE_CONFIG}" \
  --device cuda:0 \
  > "${OUTPUT_ROOT}/logs/capture.log" 2>&1

date -u +%FT%TZ > "${OUTPUT_ROOT}/complete"
