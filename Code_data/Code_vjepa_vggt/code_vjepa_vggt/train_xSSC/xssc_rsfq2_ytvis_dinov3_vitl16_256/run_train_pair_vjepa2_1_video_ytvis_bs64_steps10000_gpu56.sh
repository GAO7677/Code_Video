#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-5,6}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-xssc_vjepa2_1_video}"
DATA_DIR="${DATA_DIR:-/data/gaoya/dataset}"

NONCAUSAL_SAVE_DIR="${NONCAUSAL_SAVE_DIR:-/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_noncausal_ytvis_hq_bs64_steps10000}"
CAUSAL_SAVE_DIR="${CAUSAL_SAVE_DIR:-/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_prefix_causal_ytvis_hq_bs64_steps10000}"

run_stage() {
  local label="$1"
  local launcher="$2"
  local save_dir="$3"
  echo "[train-pair] start label=${label} gpus=${GPU_IDS} utc=$(date -u +%FT%TZ)"
  env \
    GPU_IDS="${GPU_IDS}" \
    NPROC_PER_NODE=2 \
    SEED="${SEED}" \
    SAVE_DIR="${save_dir}" \
    DATA_DIR="${DATA_DIR}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_MODE=online \
    bash "${ROOT}/${launcher}"
  echo "[train-pair] complete label=${label} utc=$(date -u +%FT%TZ)"
}

run_stage \
  noncausal \
  run_train_rsfq2_ytvis_hq_vjepa2_1_vitl16_256_video_slot512.sh \
  "${NONCAUSAL_SAVE_DIR}"

run_stage \
  prefix_causal \
  run_train_rsfq2_ytvis_hq_vjepa2_1_vitl16_256_video_prefix_causal.sh \
  "${CAUSAL_SAVE_DIR}"

