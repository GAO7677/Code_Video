#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage1_query_time_validation/runs
WAN_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
ANALYZER=/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics/analyze_query_time_head_validation.py
RENDERER=/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics/render_query_time_validation_overlays.py
EXPECTED_RUNS=15

while true; do
  completed=$(find "$RUNS_ROOT" -mindepth 3 -maxdepth 3 -type f -name complete.json | wc -l)
  errors=$(find "$RUNS_ROOT" -mindepth 3 -maxdepth 3 -type f -name error.txt -size +0c | wc -l)
  printf '[%s] complete=%s/%s errors=%s\n' "$(date -u +%FT%TZ)" "$completed" "$EXPECTED_RUNS" "$errors"
  if (( errors > 0 )); then
    printf 'Stage 1 stopped because one or more runs failed.\n' >&2
    exit 1
  fi
  if (( completed == EXPECTED_RUNS )); then
    break
  fi
  if (( completed > EXPECTED_RUNS )); then
    printf 'Stage 1 has more complete markers than expected.\n' >&2
    exit 1
  fi
  sleep 30
done

"$WAN_PYTHON" "$ANALYZER" --require-runs "$EXPECTED_RUNS"
exec "$WAN_PYTHON" "$RENDERER"
