#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU SEED [SEED ...]}"
shift
[[ "$#" -gt 0 ]] || { echo "at least one tube seed is required" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-3]$ ]] || { echo "GPU must be one of 0,1,2,3" >&2; exit 2; }

ROOT="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
MANIFEST="/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/cases_001460_5seeds.json"
OUTPUT_BASE="/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326"
FIXED_ROOT="${OUTPUT_BASE}/attention_matrix_ablations_v2"
TUBE_ROOT="${OUTPUT_BASE}/attention_matrix_ablations_temporal_tube_v1"
LOG_ROOT="${OUTPUT_BASE}/multiseed_logs"
CASE="0613pybullet_sample_001460_w002"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/gpu${GPU}.log") 2>&1
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${ROOT}"

echo "[$(date -u +%FT%TZ)] GPU${GPU} start tube seeds: $*"
for SEED in "$@"; do
  "${PYTHON}" -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py \
    --case "${CASE}" \
    --seed "${SEED}" \
    --manifest-path "${MANIFEST}" \
    --output-root "${TUBE_ROOT}" \
    --device cuda
done

echo "[$(date -u +%FT%TZ)] GPU${GPU} start fixed-query shard ${GPU}/4"
"${PYTHON}" -u AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py \
  --worker-id "${GPU}" \
  --num-workers 4 \
  --manifest-path "${MANIFEST}" \
  --output-root "${FIXED_ROOT}" \
  --top-counts 100 \
  --object-dependent-only

echo "[$(date -u +%FT%TZ)] GPU${GPU} complete"
