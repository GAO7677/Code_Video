#!/usr/bin/env bash
# Safe inference profile for legacy/stability-v2 Stage1B checkpoints.
# Grounding dedupe is implemented in the Python provider. This wrapper adds an
# all-layer residual probe and retries pathological >3-slot cases with 3 slots.
set -euo pipefail

BASE_DIR=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box

export OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO="${OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO:-0.15}"
export OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID="${OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID:--1}"
export OBJECT_BRANCH_AUTO_FALLBACK_MAX_ACTIVE_SLOTS="${OBJECT_BRANCH_AUTO_FALLBACK_MAX_ACTIVE_SLOTS:-3}"
export OBJECT_BRANCH_AUTO_FALLBACK_TRIGGER_COUNT="${OBJECT_BRANCH_AUTO_FALLBACK_TRIGGER_COUNT:-5}"

exec bash "${BASE_DIR}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh" "$@"
