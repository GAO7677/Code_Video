#!/usr/bin/env bash
set -euo pipefail

# Full OpenVid + raw phys + Genesis mixed training for Wan2.1-1.3B.
# Default temporal recipe:
#   - variable per-sample full-video length
#   - Wan 4n+1 alignment
#   - capped at 81 frames
#
# Run:
#   sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_train_openvid_full_ctx81_384x672_lora_wan21_13b_gpu3567.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="${SCRIPT_DIR}/run_train_openvid_mixed_ctx24_384x672_lora_wan21_13b_gpu0235.sh"

export GPU_SET="${GPU_SET:-3,5,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export DATASET_CONFIG="${DATASET_CONFIG:-${SCRIPT_DIR}/dataset_mix_config_openvid_full_ctx81.json}"
export TRAIN_NUM_FRAMES="${TRAIN_NUM_FRAMES:-81}"
export SAMPLE_FULL_VIDEO_MAX_FRAMES="${SAMPLE_FULL_VIDEO_MAX_FRAMES:-81}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/openvid_full_ctx81_384x672_lora}"

exec bash "${BASE_SCRIPT}"
