#!/usr/bin/env bash
set -u

ROOT=/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_case001460
LOG="$ROOT/monitor_backtrack3.log"

while true; do
  {
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    ps -eo pid,etimes,args \
      | rg 'backtrack3|build_object_query_frozen_trajectory_masks|apply_frozen_object_query_masks' \
      | rg -v 'rg ' || true
    printf 'complete='
    find "$ROOT/seeds" -maxdepth 2 -name backtrack3_p95_p99_complete | wc -l
    echo
  } >> "$LOG"
  sleep 30
done
