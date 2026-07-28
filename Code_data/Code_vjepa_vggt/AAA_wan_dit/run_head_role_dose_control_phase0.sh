#!/usr/bin/env bash
set -euo pipefail

# Run in foreground:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_head_role_dose_control_phase0.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
ROOT=/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control
OUTPUT="${ROOT}/head_classification"
LOG="${ROOT}/logs/phase0_raw_export.log"

mkdir -p "${ROOT}/logs"

exec "${PYTHON}" "${SCRIPT_DIR}/export_head_role_raw_features.py" \
  --capture-root /data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/capture \
  --query-root /data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds/query_maps \
  --seed-snapshot /data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/partial_analysis/snapshot_20260728T0245Z/common22_seeds.json \
  --input-list /data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/input_lists/test5_unique20.txt \
  --output-dir "${OUTPUT}" \
  2>&1 | tee "${LOG}"
