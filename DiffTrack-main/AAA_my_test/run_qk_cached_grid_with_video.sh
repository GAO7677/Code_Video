#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DATASET_TAG" >&2
  exit 2
fi

TAG="$1"
HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
DATASET_ROOT="/data/gaoya/agent-data/datasets/${TAG}_qk_grid_valid"
CACHE_ROOT="/data/gaoya/agent-data/cache/${TAG}_grounded_sam2_regions"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/${TAG}_difftrack_qk_grid_with_video"
STATUS_ROOT="/data/gaoya/agent-data/outputs/qk_cached_grid_with_video_status"

mkdir -p "${STATUS_ROOT}"
rm -f "${STATUS_ROOT}/${TAG}.complete" "${STATUS_ROOT}/${TAG}.failed"
trap 'date -u +%FT%TZ > "${STATUS_ROOT}/${TAG}.failed"' ERR

"${PYTHON}" "${HERE}/launch_physiciq_three_model_qk_49f.py" \
  --models gt stage1b lora baseline --gpus 0 1 2 3 4 \
  --dataset-root "${DATASET_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --analysis-layers 0 5 11 17 23 29 \
  --analysis-step-indices 0 10 20 29 39

"${PYTHON}" "${HERE}/rank_difftrack_qk_grid.py" \
  --result-root "${OUTPUT_ROOT}" \
  --cache-root "${CACHE_ROOT}"

date -u +%FT%TZ > "${STATUS_ROOT}/${TAG}.complete"
trap - ERR
