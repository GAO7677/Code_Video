#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEAD_CONFIG="${SCRIPT_DIR}/config_head_ablation_all_blocks_test5.sh"
# shellcheck source=/dev/null
source "${HEAD_CONFIG}"

HEAD_RUN_ROOT="${RUN_ROOT}"
EXPECTED_HEAD_CONFIGS=2160

SESSION="wan_head_ablation_test5_fill_baselines_gpu56"
BASELINE_OUTPUT_BASE="/data/gaoya/AAA_test_video/0623/test/v2v_wan_test5"
BASELINE_RUN_ROOT="${BASELINE_OUTPUT_BASE}/_baseline_fill"
GALLERY_SCRIPT="${SCRIPT_DIR}/serve_configured_head_ablation_gallery.py"

GPU5_MODELS="wan_lora physrvg"
GPU6_MODELS="xssc"
WAIT_SECONDS=5

