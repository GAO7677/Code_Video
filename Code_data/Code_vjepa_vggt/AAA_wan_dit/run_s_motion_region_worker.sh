#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: run_s_motion_region_worker.sh GPU}"
ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
STATE="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis/state"
LOGS="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis/logs"

mkdir -p "${STATE}" "${LOGS}"
rm -f "${STATE}/regions.complete" "${STATE}/regions.failed"
if CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  "${ROOT}/build_s_motion_region_caches.py" \
  --device cuda:0 \
  2>&1 | tee "${LOGS}/regions.log"; then
  touch "${STATE}/regions.complete"
else
  touch "${STATE}/regions.failed"
  exit 1
fi
