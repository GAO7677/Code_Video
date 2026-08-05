#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40_baseline_official_ti2v
SEEDS=(47326 90094 32466 35075 21890 49530)
while :; do
  ready=1
  for seed in "${SEEDS[@]}"; do
    [[ -f "$ROOT/seeds/seed_$(printf '%06d' "$seed")/complete" ]] || ready=0
  done
  [[ "$ready" -eq 1 ]] && break
  sleep 30
done
ALIGNMENT_MODEL=baseline OBJECT_STEP_ALIGNMENT_ROOT="$ROOT" \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/analyze_object_query_step_alignment.py
