#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/data/gaoya/miniconda3/envs/wan/bin/python"
OUTPUT_BASE="/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326"
FIXED_ROOT="${OUTPUT_BASE}/attention_matrix_ablations_v2"
TUBE_ROOT="${OUTPUT_BASE}/attention_matrix_ablations_temporal_tube_v1"
CASE="0613pybullet_sample_001460_w002"
SEEDS=(90094 68613 35075 32466 13248)

mkdir -p "${OUTPUT_BASE}/multiseed_logs"
exec > >(tee -a "${OUTPUT_BASE}/multiseed_logs/similarity_waiter.log") 2>&1
cd "${ROOT}"

while true; do
  READY=1
  STATUS=()
  for SEED in "${SEEDS[@]}"; do
    FIXED_COUNT=$(find "${FIXED_ROOT}/${CASE}/seed_$(printf '%05d' "${SEED}")" \
      -mindepth 2 -maxdepth 2 -name complete.json 2>/dev/null | wc -l)
    TUBE_COUNT=$(find "${TUBE_ROOT}/${CASE}/seed_$(printf '%05d' "${SEED}")" \
      -mindepth 2 -maxdepth 2 -name complete.json 2>/dev/null | wc -l)
    STATUS+=("${SEED}:fixed=${FIXED_COUNT}/24,tube=${TUBE_COUNT}/24")
    if [[ "${FIXED_COUNT}" -ne 24 || "${TUBE_COUNT}" -ne 24 ]]; then
      READY=0
    fi
  done
  echo "[$(date -u +%FT%TZ)] ${STATUS[*]}"
  [[ "${READY}" -eq 1 ]] && break
  sleep 60
done

for SEED in "${SEEDS[@]}"; do
  "${PYTHON}" -u AAA_my_test/analyze_legacy_ti2v_object_ablation_video_similarity.py \
    --case "${CASE}" --seed "${SEED}" --workers 6
done

echo "[$(date -u +%FT%TZ)] all five similarity reports complete"
