#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STRUCTURE_ABLATION_TYPE="${STRUCTURE_ABLATION_TYPE:-none}"
export ABLATION_TAG="${ABLATION_TAG:-structure_ablation_no_stage1a_init}"
export WANDB_NAME="${WANDB_NAME:-${ABLATION_TAG}_gpu0235}"
export NO_STAGE1A_INIT=1

exec "${SCRIPT_DIR}/structure_ablation_base_gpu0235.sh"
