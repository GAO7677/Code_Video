#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-001000 001500 002000 002500 003000 003500}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/agent-data/outputs/AAA_physv/scheme_d_v3_object_tube_checkpoint_val_20260715}"

for checkpoint_step in ${CHECKPOINT_STEPS}; do
  CHECKPOINT_STEP="${checkpoint_step}" \
  OUTPUT_ROOT="${OUTPUT_BASE}/step-${checkpoint_step}" \
  bash "${PROJECT}/watch_checkpoint_validation.sh"
done
