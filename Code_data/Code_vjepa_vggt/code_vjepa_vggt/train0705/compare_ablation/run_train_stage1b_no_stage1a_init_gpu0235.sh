#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ABLATION_TAG="${ABLATION_TAG:-no_stage1a_init}"
export NO_STAGE1A_INIT=1
export DISABLE_COTRACKER="${DISABLE_COTRACKER:-0}"
export DISABLE_JEPA="${DISABLE_JEPA:-0}"
export DISABLE_VGGT="${DISABLE_VGGT:-0}"
export WANDB_NAME="${WANDB_NAME:-train0705_${ABLATION_TAG}_gpu0235}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705_compare_ablation/${ABLATION_TAG}}"

exec "${SCRIPT_DIR}/run_train_stage1b_compare_ablation_base_gpu0235.sh"
