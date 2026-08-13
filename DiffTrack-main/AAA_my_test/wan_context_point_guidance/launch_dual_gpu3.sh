#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-3}"
MAX_USED_MIB="${MAX_USED_MIB:-8000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/v1}"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
SCRIPT=/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/wan_context_point_guidance/run_dual_protocol.py
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/dual_gpu${GPU_ID}.log"

if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU 4 is prohibited by workspace rules." >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[queue] $(date -u +%FT%TZ) waiting for GPU ${GPU_ID}: memory.used <= ${MAX_USED_MIB} MiB"
while true; do
  used="$({ nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits || echo 999999; } | head -n1 | tr -d ' ')"
  if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
    break
  fi
  echo "[queue] $(date -u +%FT%TZ) GPU ${GPU_ID} used=${used} MiB; still waiting"
  sleep "${POLL_SECONDS}"
done

echo "[queue] $(date -u +%FT%TZ) GPU ${GPU_ID} admitted"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main:/home/gaoya/Code_Video/DiffTrack-main
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_backend() {
  local backend="$1"
  echo "[run] $(date -u +%FT%TZ) backend=${backend} stage=all"
  "${PYTHON}" -u "${SCRIPT}" \
    --backend "${backend}" \
    --stage all \
    --device cuda:0 \
    --output-root "${OUTPUT_ROOT}"
}

run_backend firstframe_ti2v
run_backend context8_v2v
echo "[complete] $(date -u +%FT%TZ) both protocols finished"
