#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 GPU SHARD_COUNT}"
SHARDS="${2:?usage: $0 GPU SHARD_COUNT}"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed090094_case001460"
RUNNER="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_attention_lora_neighbor_ranking_gpu.sh"
PROFILES="temporal_causal strict_past strict_future exclude_current context_only"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for branch in both conditional unconditional; do
  marker="pck32_priority_${branch}_gpu${GPU}.complete"
  while ! ATTENTION_NEIGHBOR_RANKING_ROOT="${ROOT}" \
      ATTENTION_RANKING_FIXED_SEED=90094 \
      ATTENTION_NEIGHBOR_CRITERIA=pck32 \
      ATTENTION_NEIGHBOR_PROFILES="${PROFILES}" \
      ATTENTION_CFG_BRANCH_MODE="${branch}" \
      ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
      ATTENTION_NEIGHBOR_COMPLETE_NAME="${marker}" \
      bash "${RUNNER}" "${GPU}" "${SHARDS}"; do
    sleep 120
  done
done
