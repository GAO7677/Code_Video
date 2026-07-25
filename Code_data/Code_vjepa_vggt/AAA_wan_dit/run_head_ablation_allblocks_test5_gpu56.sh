#!/usr/bin/env bash
set -euo pipefail

# Dry-run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_head_ablation_allblocks_test5_gpu56.sh --dry-run
# Launch:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_head_ablation_allblocks_test5_gpu56.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${HEAD_ABLATION_CONFIG:-${SCRIPT_DIR}/head_ablation_allblocks_test5_gpu56.json}"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"

exec "${PYTHON}" "${SCRIPT_DIR}/run_configured_head_ablation_sweep.py" \
  --config "${CONFIG}" "$@"
