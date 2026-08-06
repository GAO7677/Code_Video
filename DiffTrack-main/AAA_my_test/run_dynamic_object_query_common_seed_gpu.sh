#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU SEED}"
SEED="${2:?usage: $0 GPU SEED}"
DIFFTRACK=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
ROOT=/data/gaoya/agent-data/outputs/object_query_dynamic_common_case001460
CASE=0613pybullet_sample_001460_w002
SID=$(printf '%06d' "$SEED")
SEED_ROOT="$ROOT/seeds/seed_$SID"
CASE_LIST="$ROOT/case_list.txt"
REGION_CACHE=/data/gaoya/agent-data/cache/test100_51_grounded_sam2_regions/case_test100_51_048_0613pybullet_sample_001460_w002
VIDEO40=/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds/seed_$SID/original.mp4
VIDEO10=/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_10step_case001460/seeds/seed_$SID/probe_top100/videos/lora/cases/$CASE/original.mp4

mkdir -p "$ROOT/logs" "$SEED_ROOT"
printf '%s\n' "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/$CASE.json" > "$CASE_LIST"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OBJECT_QUERY_REGION_CACHE="$REGION_CACHE"
export QK_ATTENTION_CAPTURE_CASE="$CASE"
cd "$DIFFTRACK"

if [[ ! -f "$SEED_ROOT/tracks_manifest.json" ]]; then
  "$PYTHON" AAA_my_test/prepare_dynamic_object_query_tracks.py \
    --seed "$SEED" --video40 "$VIDEO40" --video10 "$VIDEO10" \
    --region-cache "$REGION_CACHE" --output-root "$SEED_ROOT"
fi

run_schedule() {
  local steps="$1" label="steps$1" video="$2" expected
  local schedule_root="$SEED_ROOT/$label"
  local track_file="$schedule_root/dynamic_query_tracks.npz"
  local capture_root="$schedule_root/captures"
  local render_root="$schedule_root/attention_overlays"
  local video_root="$schedule_root/inference"
  expected=$((steps * 2))
  mkdir -p "$capture_root" "$render_root" "$video_root"
  local count
  count=$(find "$capture_root" -maxdepth 1 -name 'step_*.npz' 2>/dev/null | wc -l)
  if [[ "$count" -lt "$expected" ]]; then
    "$PYTHON" AAA_my_test/run_dynamic_object_query_common_worker.py \
      --seed "$SEED" --inference-steps "$steps" --track-file "$track_file" \
      --capture-root "$capture_root" --input-json-list "$CASE_LIST" --output-root "$video_root"
  fi
  "$PYTHON" AAA_my_test/render_dynamic_object_query_common.py \
    --capture-root "$capture_root" --render-root "$render_root" --video "$video"
  count=$(find "$capture_root" -maxdepth 1 -name 'step_*.npz' 2>/dev/null | wc -l)
  [[ "$count" -ge "$expected" ]] || {
    echo "Incomplete dynamic ${steps}-step capture: $count/$expected" >&2
    exit 1
  }
}

run_schedule 40 "$VIDEO40"
run_schedule 10 "$VIDEO10"
printf 'gpu=%s\nseed=%s\nquery=CoTracker_dynamic_per_latent_frame\ncompleted=%s\n' \
  "$GPU" "$SEED" "$(date -u +%FT%TZ)" > "$SEED_ROOT/complete"
