#!/usr/bin/env bash
# Scheme C: no no-object teacher on physics data; retain it for OpenVid replay.
set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
BASE_LAUNCHER="${ROOT}/run_train_stage1b_raw49f_kubric_openvid_replay_preserve_init3500_gpu012356.sh"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

PYBULLET_TEACHER_PRESERVATION_WEIGHT=0.0 \
KUBRIC_TEACHER_PRESERVATION_WEIGHT=0.0 \
OPENVID_TEACHER_PRESERVATION_WEIGHT=0.05 \
TEACHER_PRESERVATION_WEIGHT=0.05 \
RUN_TAG="${RUN_TAG}" \
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_replay_preserve_scheme_c_init3500_${RUN_TAG}}" \
WANDB_NAME="${WANDB_NAME:-stage1b_raw49f_replay_preserve_scheme_c_init3500_${RUN_TAG}}" \
bash "${BASE_LAUNCHER}"
