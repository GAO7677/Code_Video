#!/usr/bin/env bash
set -euo pipefail
ROOT="/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed090094_case001460"
SCRIPT="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_attention_lora_neighbor_ranking_gpu.sh"
GPU="${1:?usage: $0 GPU_ID}"
for shard in 0 1 2 3 4; do
  while [[ ! -f "${ROOT}/logs/gpu${shard}.complete" ]]; do sleep 60; done
done
run_branch() {
  local branch="$1" profiles="$2" marker="$3"
  while ! ATTENTION_NEIGHBOR_RANKING_ROOT="${ROOT}" ATTENTION_RANKING_FIXED_SEED=90094 ATTENTION_CFG_BRANCH_MODE="${branch}" ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 ATTENTION_NEIGHBOR_PROFILES="${profiles}" ATTENTION_NEIGHBOR_COMPLETE_NAME="${marker}" bash "${SCRIPT}" "${GPU}" 5; do sleep 120; done
}
run_branch both "exclude_current context_only" "both_new_masks_gpu${GPU}.complete"
run_branch conditional "" "conditional_gpu${GPU}.complete"
run_branch unconditional "" "unconditional_gpu${GPU}.complete"
