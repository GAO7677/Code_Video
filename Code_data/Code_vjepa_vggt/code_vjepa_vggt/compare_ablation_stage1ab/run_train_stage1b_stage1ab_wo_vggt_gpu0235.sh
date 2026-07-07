#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STRUCTURE_ABLATION_TYPE=wo_vggt
export ABLATION_TAG="${ABLATION_TAG:-stage1b_wo_vggt}"
export WANDB_NAME="${WANDB_NAME:-${ABLATION_TAG}_gpu0235}"

exec "${SCRIPT_DIR}/run_train_stage1b_structure_ablation_base_gpu0235.sh" "$@"
