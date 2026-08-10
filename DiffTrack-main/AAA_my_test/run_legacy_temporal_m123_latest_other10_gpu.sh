#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 GPU_ID WORKER_ID [NUM_WORKERS]" >&2
  exit 2
fi

GPU_ID=$1
WORKER_ID=$2
NUM_WORKERS=${3:-4}
REPO=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BASE=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326
MANIFEST=$BASE/cases_other10_6seeds_latest.json
HEAD_RANKING=$BASE/pck_head_scopes_s039_latest2735.json
OUTPUT_ROOT=$BASE/attention_matrix_ablations_temporal_tube_v1
LOG_ROOT=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/logs/m123_latest_other10_s039r2735

mkdir -p "$LOG_ROOT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONUNBUFFERED=1

echo "[$(date -u +%FT%TZ)] GPU=$GPU_ID worker=$WORKER_ID/$NUM_WORKERS manifest=$MANIFEST scopes=top100,bottom100 ranking=s039r2735"
exec "$PYTHON" -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py \
  --all-samples \
  --manifest-path "$MANIFEST" \
  --head-ranking-path "$HEAD_RANKING" \
  --ranking-tag s039r2735 \
  --worker-id "$WORKER_ID" \
  --num-workers "$NUM_WORKERS" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --head-scopes top100 bottom100 \
  --mask-modes \
    self_only self_same self_future self_past \
    incoming_only incoming_same incoming_future incoming_past \
    outgoing_only outgoing_same outgoing_future outgoing_past \
  2>&1 | tee -a "$LOG_ROOT/gpu${GPU_ID}_worker${WORKER_ID}.log"
