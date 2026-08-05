#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40
for seed in 047326 090094 032466 035075 021890 049530; do
  while [[ ! -f "$ROOT/seeds/seed_$seed/complete" ]]; do
    sleep 30
  done
done
cd /home/gaoya/Code_Video/DiffTrack-main
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  AAA_my_test/analyze_object_query_step_alignment.py --root "$ROOT" \
  > "$ROOT/analysis.log" 2>&1
printf 'completed=%s\n' "$(date -u +%FT%TZ)" > "$ROOT/analysis/complete"
