#!/usr/bin/env bash
# Scheme C plus video-local entity-ID hard routing from prompt spans to tracked slots.
set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
SCHEME_C_LAUNCHER="${ROOT}/run_train_stage1b_raw49f_replay_preserve_scheme_c_init3500_gpu012356.sh"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

TRAIN_SCRIPT="${ROOT}/train_stage1b_no_gt_box_replay_preserve_entity_id_binding.py" \
RUN_TAG="${RUN_TAG}" \
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_scheme_c_entity_id_binding_init3500_${RUN_TAG}}" \
WANDB_NAME="${WANDB_NAME:-stage1b_raw49f_scheme_c_entity_id_binding_init3500_${RUN_TAG}}" \
bash "${SCHEME_C_LAUNCHER}"
