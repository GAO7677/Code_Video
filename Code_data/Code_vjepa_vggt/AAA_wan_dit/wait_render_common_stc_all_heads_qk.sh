#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/wait_render_common_stc_all_heads_qk.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
ROOT=/data/gaoya/agent-data/outputs/wan_dit_common_stc_all_heads_qk_seed851
GALLERY=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/common-stc-all-heads-qk-seed851

while true; do
  done_count="$(find "${ROOT}/state" -maxdepth 1 -name '*.complete' | wc -l)"
  failed_count="$(find "${ROOT}/state" -maxdepth 1 -name '*.failed' | wc -l)"
  echo "[common-stc-qk] capture ${done_count}/3 complete, ${failed_count} failed"
  if (( failed_count > 0 )); then
    exit 1
  fi
  if (( done_count == 3 )); then
    break
  fi
  sleep 60
done

"${PYTHON}" "${SCRIPT_DIR}/render_selected_qk_batch.py" \
  --selection "${ROOT}/selection.json" \
  --capture-root "${ROOT}/capture" --output-dir "${ROOT}/heatmaps" \
  2>&1 | tee "${ROOT}/logs/render.log"
"${PYTHON}" "${SCRIPT_DIR}/build_common_stc_all_heads_qk_gallery.py" \
  --heads "${ROOT}/heads.csv" --selection "${ROOT}/selection.json" \
  --heatmap-root "${ROOT}/heatmaps" --output-dir "${GALLERY}" \
  2>&1 | tee "${ROOT}/logs/gallery.log"
touch "${ROOT}/state/gallery.complete"
