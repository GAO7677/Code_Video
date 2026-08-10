#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_ID WORKER_ID" >&2
  exit 2
fi

GPU_ID=$1
WORKER_ID=$2
ROOT=/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test

echo "[$(date -u +%FT%TZ)] phase 1/2: pilot Bottom100 + All720"
"$ROOT/run_legacy_temporal_m123_head_scopes_001460_gpu.sh" \
  "$GPU_ID" "$WORKER_ID" 4
echo "[$(date -u +%FT%TZ)] phase 2/2: latest other10 Top100 + Bottom100"
exec "$ROOT/run_legacy_temporal_m123_latest_other10_gpu.sh" \
  "$GPU_ID" "$WORKER_ID" 4
