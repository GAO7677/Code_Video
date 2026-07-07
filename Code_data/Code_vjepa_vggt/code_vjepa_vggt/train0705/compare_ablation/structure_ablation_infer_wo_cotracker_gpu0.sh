#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STRUCTURE_ABLATION_TYPE=wo_cotracker

if [[ -z "${CHECKPOINT:-}" ]]; then
  export CHECKPOINT="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train0705_ablation/structure_ablation_wo_cotracker_smoke_train/checkpoints/step-000001"
fi
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  export OUTPUT_ROOT="/data/gaoya/agent-data/outputs/train0705_ablation_tmp/structure_ablation_wo_cotracker"
fi

exec "${SCRIPT_DIR}/structure_ablation_infer_base_gpu0.sh"
