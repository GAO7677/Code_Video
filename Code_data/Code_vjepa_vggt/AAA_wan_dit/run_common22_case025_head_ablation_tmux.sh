#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_common22_case025_head_ablation_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/common22_public_head_ablation_case025.json"
SESSION=wan_common22_case025_head_ablation
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_public_head_ablation_case025
CONFIG="${CONFIG}" SESSION="${SESSION}" \
  bash "${SCRIPT_DIR}/run_common22_public_head_ablation_tmux.sh"
