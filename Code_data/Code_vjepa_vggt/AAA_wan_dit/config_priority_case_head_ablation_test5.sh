#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEAD_CONFIG="${SCRIPT_DIR}/config_head_ablation_all_blocks_test5.sh"
# shellcheck source=/dev/null
source "${HEAD_CONFIG}"

SESSION="wan_head_priority_case001460_blocks0511171929_gpu56"
PRIORITY_RUN_ROOT="${OUTPUT_BASE}/_priority_case"
PRIORITY_INPUT_JSON="/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_001460_w002.json"
PRIORITY_BLOCKS="5 11 17 19 29"
PRIORITY_HEADS="0-23"
PRIORITY_MODELS="wan_lora xssc physrvg"
PRIORITY_GPUS="5 6"
EXPECTED_PRIORITY_TASKS=360
PRIORITY_GPU_MAX_USED_MIB=8000
PRIORITY_GPU_WAIT_SECONDS=5
GALLERY_SCRIPT="${SCRIPT_DIR}/serve_configured_head_ablation_gallery.py"

