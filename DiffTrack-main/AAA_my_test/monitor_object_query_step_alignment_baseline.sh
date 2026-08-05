#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40_baseline_official_ti2v
mkdir -p "$ROOT"
while :; do
  {
    printf '%s' "$(date -u +%FT%TZ)"
    for seed in 47326 90094 32466 35075 21890 49530; do
      sid=$(printf '%06d' "$seed")
      n40=$(find "$ROOT/seeds/seed_$sid/steps40/captures" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
      n10=$(find "$ROOT/seeds/seed_$sid/steps10/captures" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
      printf ' seed_%s=%s/80,%s/20' "$sid" "$n40" "$n10"
    done
    printf '\n'
  } >> "$ROOT/monitor.log"
  sleep 30
done
