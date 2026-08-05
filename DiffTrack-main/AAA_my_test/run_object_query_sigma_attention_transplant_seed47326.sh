#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-2}"
ROOT=/data/gaoya/agent-data/outputs/object_query_sigma_attention_transplant_case001460/seed_047326
DIFFTRACK=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER=$DIFFTRACK/AAA_my_test/run_object_query_reverse_attention_transplant_worker.py
RENDERER=$DIFFTRACK/AAA_my_test/render_reverse_transplant_top100_mean.py
CASE=0613pybullet_sample_001460_w002
CASE_JSON=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/$CASE.json
CASE_LIST=$ROOT/case_list.txt
MAPPING=/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40/analysis/reverse/per_head_best_matches_by_sample_reverse.csv
REGION_CACHE=/data/gaoya/agent-data/cache/test100_51_grounded_sam2_regions/case_test100_51_048_0613pybullet_sample_001460_w002
DONOR_VIDEO_ROOT=$ROOT/donor_run
REPLACEMENT_VIDEO_ROOT=$ROOT/replacement_run
DONOR_ROOT=$ROOT/donor_rows
CAPTURE_ROOT=$ROOT/replacement_captures
OVERLAY_ROOT=$ROOT/replacement_overlays
BASELINE10=/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_10step_case001460/seeds/seed_047326/probe_top100/videos/lora/cases/$CASE/original.mp4
BASELINE40=/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds/seed_047326/original.mp4

mkdir -p "$ROOT/logs" "$DONOR_ROOT" "$CAPTURE_ROOT" "$OVERLAY_ROOT"
printf '%s\n' "$CASE_JSON" > "$CASE_LIST"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OBJECT_QUERY_REGION_CACHE="$REGION_CACHE"
export QK_ATTENTION_CAPTURE_CASE="$CASE"
cd "$DIFFTRACK"

link_original() {
  local source="$1" run_root="$2"
  local target="$run_root/lora/cases/$CASE/original.mp4"
  mkdir -p "$(dirname "$target")"
  [[ -e "$target" ]] || ln -s "$source" "$target"
}

link_original "$BASELINE10" "$DONOR_VIDEO_ROOT"
"$PYTHON" "$WORKER" --mode donor --seed 47326 \
  --donor-root "$DONOR_ROOT" --mapping-csv "$MAPPING" --capture-root "$CAPTURE_ROOT" \
  --input-json-list "$CASE_LIST" --output-root "$DONOR_VIDEO_ROOT"

link_original "$BASELINE40" "$REPLACEMENT_VIDEO_ROOT"
"$PYTHON" "$WORKER" --mode sigma_replacement --seed 47326 \
  --donor-root "$DONOR_ROOT" --mapping-csv "$MAPPING" --capture-root "$CAPTURE_ROOT" \
  --input-json-list "$CASE_LIST" --output-root "$REPLACEMENT_VIDEO_ROOT"

VIDEO=$REPLACEMENT_VIDEO_ROOT/lora/cases/$CASE/top100_steps_00_40.mp4
"$PYTHON" "$RENDERER" --capture-root "$CAPTURE_ROOT" --video "$VIDEO" --output-root "$OVERLAY_ROOT"

printf 'seed=47326\ngpu=%s\nmatching=nearest_log_sigma\ndonor_files=%s\ncapture_files=%s\ncompleted=%s\n' \
  "$GPU" \
  "$(find "$DONOR_ROOT" -maxdepth 1 -name 'step_*.npz' | wc -l)" \
  "$(find "$CAPTURE_ROOT" -maxdepth 1 -name 'step_*.npz' | wc -l)" \
  "$(date -u +%FT%TZ)" > "$ROOT/complete"
