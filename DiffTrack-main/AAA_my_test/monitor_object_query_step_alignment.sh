#!/usr/bin/env bash
set -u

ROOT=/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40
mkdir -p "$ROOT"
while true; do
  {
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    ps -eo pid,etimes,args | rg 'object_query_step_alignment|step10_vs_step40' | rg -v 'rg ' || true
    printf 'completed_seeds='
    find "$ROOT/seeds" -maxdepth 2 -name complete 2>/dev/null | wc -l
    echo
  } >> "$ROOT/monitor.log"
  sleep 30
done
