#!/usr/bin/env bash
set -u

ROOT=/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_10step_case001460
LOG="$ROOT/monitor.log"
mkdir -p "$ROOT"
while true; do
  {
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    ps -eo pid,etimes,args \
      | rg 'frozen_trajectory_10step|run_object_query_frozen_trajectory_10step' \
      | rg -v 'rg ' || true
    printf 'completed_seeds='
    find "$ROOT/seeds" -maxdepth 2 -name complete 2>/dev/null | wc -l
    echo
  } >> "$LOG"
  sleep 30
done
