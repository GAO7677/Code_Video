#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: run_s_motion_incremental_worker.sh GPU SHARD_ID NUM_SHARDS SNAPSHOT_DIR}"
SHARD_ID="${2:?usage: run_s_motion_incremental_worker.sh GPU SHARD_ID NUM_SHARDS SNAPSHOT_DIR}"
NUM_SHARDS="${3:?usage: run_s_motion_incremental_worker.sh GPU SHARD_ID NUM_SHARDS SNAPSHOT_DIR}"
SNAPSHOT_DIR="${4:?usage: run_s_motion_incremental_worker.sh GPU SHARD_ID NUM_SHARDS SNAPSHOT_DIR}"

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
OUTPUT="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis"
STATE="${SNAPSHOT_DIR}/state"
LOGS="${SNAPSHOT_DIR}/logs"
LABEL="shard_${SHARD_ID}_gpu${GPU}"

mkdir -p "${STATE}" "${LOGS}"
rm -f "${STATE}/${LABEL}.complete" "${STATE}/${LABEL}.failed"
touch "${STATE}/${LABEL}.running"
trap 'rm -f "${STATE}/${LABEL}.running"' EXIT

if CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  "${ROOT}/extract_s_motion_features.py" \
  --inventory "${SNAPSHOT_DIR}/inventory.json" \
  --output-root "${OUTPUT}" \
  --device cuda:0 \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  2>&1 | tee "${LOGS}/${LABEL}.log"; then
  touch "${STATE}/${LABEL}.complete"
else
  touch "${STATE}/${LABEL}.failed"
  exit 1
fi
