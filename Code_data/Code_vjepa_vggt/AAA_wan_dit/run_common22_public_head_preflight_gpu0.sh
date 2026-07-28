#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_common22_public_head_preflight_gpu0.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_public_head_ablation_preflight
INPUT_LIST="${SCRIPT_DIR}/common22_public_head_ablation_preflight_case.txt"
REPORT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/partial_analysis/snapshot_20260728T0245Z/common22/aggregate_heads.csv

for model in wan_lora xssc physrvg; do
  MODEL="${model}" SEED=851 ROLE=T GPU=0 INPUT_LIST="${INPUT_LIST}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" PUBLIC_HEAD_REPORT="${REPORT}" \
    bash "${SCRIPT_DIR}/run_common22_public_head_ablation_job.sh" \
    2>&1 | tee "${OUTPUT_ROOT}_${model}.log"
done
