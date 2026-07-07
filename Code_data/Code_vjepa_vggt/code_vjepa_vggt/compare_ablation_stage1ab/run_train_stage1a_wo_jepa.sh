#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STRUCTURE_ABLATION_TYPE=wo_jepa
export ABLATION_TAG="${ABLATION_TAG:-stage1a_wo_jepa}"
export WANDB_NAME="${WANDB_NAME:-${ABLATION_TAG}}"

exec "${SCRIPT_DIR}/run_train_stage1a_structure_ablation_base.sh" "$@"
