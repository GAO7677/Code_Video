#!/usr/bin/env bash
# Usage:
# GPU=5 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_stc_motion_analysis_once.sh

set -euo pipefail

GPU="${GPU:-5}"
ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/wan_dit_stc_motion_analysis"
CONFIG="${ROOT}/common22_public_head_ablation_case025_with_extremes.json"

"${PYTHON}" "${ROOT}/build_common22_public_head_multiseed_gallery.py" \
  --config "${CONFIG}"
"${PYTHON}" "${ROOT}/build_stc_motion_inventory.py" \
  --output-root "${OUTPUT_ROOT}"
"${PYTHON}" "${ROOT}/extract_stc_motion_features.py" \
  --inventory "${OUTPUT_ROOT}/inventory.json" \
  --output-root "${OUTPUT_ROOT}" \
  --device "cuda:${GPU}"
"${PYTHON}" "${ROOT}/analyze_stc_motion.py" \
  --inventory "${OUTPUT_ROOT}/inventory.json" \
  --output-root "${OUTPUT_ROOT}" \
  --minimum-seeds 3

echo "[motion-analysis] http://127.0.0.1:8944/multiseed/motion-analysis/"
