#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: run_s_motion_feature_worker.sh GPU SHARD_ID NUM_SHARDS}"
SHARD_ID="${2:?usage: run_s_motion_feature_worker.sh GPU SHARD_ID NUM_SHARDS}"
NUM_SHARDS="${3:?usage: run_s_motion_feature_worker.sh GPU SHARD_ID NUM_SHARDS}"
ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
STATE="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis/state"
LOGS="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis/logs"
LABEL="features_shard_${SHARD_ID}"
MIN_FREE_MIB="${S_MOTION_MIN_FREE_MIB:-16000}"

mkdir -p "${STATE}" "${LOGS}"
rm -f "${STATE}/${LABEL}.complete" "${STATE}/${LABEL}.failed"
while true; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="${GPU}" | tr -d ' ')"
  if [[ "${free_mib}" -ge "${MIN_FREE_MIB}" ]]; then
    break
  fi
  echo "[${LABEL}] waiting for GPU${GPU}: free=${free_mib} MiB, required=${MIN_FREE_MIB} MiB"
  sleep 30
done
if CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  "${ROOT}/extract_s_motion_features.py" \
  --device cuda:0 \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  2>&1 | tee "${LOGS}/${LABEL}_gpu${GPU}.log"; then
  touch "${STATE}/${LABEL}.complete"
else
  touch "${STATE}/${LABEL}.failed"
  exit 1
fi
