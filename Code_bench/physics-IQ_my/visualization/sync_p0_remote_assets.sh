#!/usr/bin/env bash
set -euo pipefail

# Keep the remote-only P0 submission under /data; the dashboard uses symlinks
# and never copies large media into the code repository.
CACHE_ROOT="/data/gaoya/agent-data/cache/physics-iq-verified/remote/xssc-loss-dinov3"
REMOTE_ROOT="/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified"
RUN_NAME="full_sa_no_object_xssc_loss_dinov3_movic_step50000-step-000500-2c970f718bcf-bpp-run_01"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$CACHE_ROOT/submission" "$CACHE_ROOT/evaluation"
rsync -a "118:$REMOTE_ROOT/generated_videos_5s/$RUN_NAME/" "$CACHE_ROOT/submission/"
rsync -a "118:$REMOTE_ROOT/evaluation/physics-IQ-benchmark-verified/results/$RUN_NAME.csv" \
  "$CACHE_ROOT/evaluation/"
rsync -a "118:$REMOTE_ROOT/evaluation/physics-IQ-benchmark-verified/results/${RUN_NAME}_metrics.json" \
  "$CACHE_ROOT/evaluation/"

python3 "$SCRIPT_DIR/build_strict_dashboard.py"
