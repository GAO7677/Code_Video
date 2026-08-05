#!/usr/bin/env bash
set -u -o pipefail

GPU="${1:?usage: $0 GPU SEED...}"
shift
ROOT=/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40
mkdir -p "$ROOT/logs"
cd /home/gaoya/Code_Video/DiffTrack-main
for seed in "$@"; do
  sid=$(printf '%06d' "$seed")
  [[ -f "$ROOT/seeds/seed_$sid/complete" ]] && continue
  log="$ROOT/logs/gpu${GPU}_seed${seed}.log"
  success=0
  for attempt in 1 2 3; do
    printf '%s gpu=%s seed=%s attempt=%s start_alignment_capture\n' \
      "$(date -u +%FT%TZ)" "$GPU" "$seed" "$attempt" | tee -a "$log"
    if ./AAA_my_test/run_object_query_step_alignment_capture_seed_gpu.sh \
      "$GPU" "$seed" 2>&1 | tee -a "$log"; then
      success=1
      break
    fi
    sleep 60
  done
  [[ "$success" == 1 ]] || exit 1
done
