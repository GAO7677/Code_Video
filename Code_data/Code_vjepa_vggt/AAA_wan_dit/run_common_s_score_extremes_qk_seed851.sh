#!/usr/bin/env bash
set -euo pipefail

# GPU=6 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_common_s_score_extremes_qk_seed851.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
GPU="${GPU:-6}"
SEED=851
REPORT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/partial_analysis/snapshot_20260728T0245Z/common22/aggregate_heads.csv
INPUT_LIST="${SCRIPT_DIR}/common22_public_head_ablation_case025.txt"
ROOT=/data/gaoya/agent-data/outputs/wan_dit_common_s_score_extremes_qk_seed851
GALLERY=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/s_score_extremes_seed851

mkdir -p "${ROOT}/logs"
"${PYTHON}" "${SCRIPT_DIR}/build_common_s_score_extremes_selection.py" \
  --report "${REPORT}" --input-list "${INPUT_LIST}" \
  --output-dir "${ROOT}" --count 10 --seed "${SEED}"

for model in wan_lora xssc physrvg; do
  MODEL="${model}" GPU="${GPU}" SEED="${SEED}" ROOT="${ROOT}" \
    OUTPUT_ROOT="${ROOT}/capture/${model}" \
    SELECTION="${ROOT}/selection.json" INPUT_LIST="${INPUT_LIST}" \
    STEPS=5,15,25,35 OUTPUT_BINS=512 QUERY_CHUNK=64 \
    bash "${SCRIPT_DIR}/run_selected_qk_capture.sh" \
    2>&1 | tee "${ROOT}/logs/${model}.log"
done

"${PYTHON}" "${SCRIPT_DIR}/render_selected_qk_batch.py" \
  --selection "${ROOT}/selection.json" \
  --capture-root "${ROOT}/capture" --output-dir "${ROOT}/heatmaps"
"${PYTHON}" "${SCRIPT_DIR}/build_common_s_score_extremes_qk_gallery.py" \
  --ranking "${ROOT}/ranking.csv" --heatmap-root "${ROOT}/heatmaps" \
  --output-dir "${GALLERY}" --seed "${SEED}"
