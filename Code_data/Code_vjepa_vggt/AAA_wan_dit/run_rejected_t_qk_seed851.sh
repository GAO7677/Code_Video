#!/usr/bin/env bash
set -euo pipefail

# GPU=5 MODEL=xssc bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_rejected_t_qk_seed851.sh
# SKIP_CAPTURE=1 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_rejected_t_qk_seed851.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
GPU="${GPU:-5}"
MODEL="${MODEL:-all}"
SEED=851
REPORT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/partial_analysis/snapshot_20260728T0245Z/common22/aggregate_heads.csv
INPUT_LIST="${SCRIPT_DIR}/common22_public_head_ablation_case025.txt"
ROOT=/data/gaoya/agent-data/outputs/wan_dit_rejected_t_qk_seed851
GALLERY=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/rejected-t-heads-qk-seed851

if [[ "${GPU}" == "4" ]]; then
  echo "GPU 4 is disabled by workspace policy" >&2
  exit 2
fi
case "${MODEL}" in
  all|wan_lora|xssc|physrvg) ;;
  *) echo "unsupported MODEL=${MODEL}" >&2; exit 2 ;;
esac

mkdir -p "${ROOT}/logs"
"${PYTHON}" "${SCRIPT_DIR}/build_rejected_t_qk_selection.py" \
  --report "${REPORT}" --input-list "${INPUT_LIST}" \
  --output-dir "${ROOT}" --seed "${SEED}" --per-group 2

if [[ "${SKIP_CAPTURE:-0}" != "1" ]]; then
  if [[ "${MODEL}" == "all" ]]; then
    models=(wan_lora xssc physrvg)
  else
    models=("${MODEL}")
  fi
  for model in "${models[@]}"; do
    MODEL="${model}" GPU="${GPU}" SEED="${SEED}" ROOT="${ROOT}" \
      OUTPUT_ROOT="${ROOT}/capture/${model}" \
      SELECTION="${ROOT}/selection.json" INPUT_LIST="${INPUT_LIST}" \
      STEPS=5,15,25,35 OUTPUT_BINS=512 QUERY_CHUNK=64 \
      bash "${SCRIPT_DIR}/run_selected_qk_capture.sh" \
      2>&1 | tee "${ROOT}/logs/${model}.log"
  done
fi

render_args=(
  --selection "${ROOT}/selection.json"
  --capture-root "${ROOT}/capture"
  --output-dir "${ROOT}/heatmaps"
)
if [[ "${MODEL}" != "all" ]]; then
  render_args+=(--models "${MODEL}")
fi
"${PYTHON}" "${SCRIPT_DIR}/render_selected_qk_batch.py" "${render_args[@]}"
"${PYTHON}" "${SCRIPT_DIR}/build_rejected_t_qk_gallery.py" \
  --heads "${ROOT}/heads.csv" --selection "${ROOT}/selection.json" \
  --heatmap-root "${ROOT}/heatmaps" --output-dir "${GALLERY}"
