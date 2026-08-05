#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
ROOT=/data/gaoya/agent-data/outputs/object_query_similarity_delta_mask1x1_case001460/seed_047326
DIFFTRACK=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER=$DIFFTRACK/AAA_my_test/run_object_query_reverse_attention_transplant_worker.py
RENDERER=$DIFFTRACK/AAA_my_test/render_positive_delta_mask_top100_mean.py
CASE=0613pybullet_sample_001460_w002
CASE_JSON=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/$CASE.json
CASE_LIST=$ROOT/case_list.txt
MAPPING=/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40/analysis/reverse/per_head_best_matches_by_sample_reverse.csv
REGION_CACHE=/data/gaoya/agent-data/cache/test100_51_grounded_sam2_regions/case_test100_51_048_0613pybullet_sample_001460_w002
DONOR_ROOT=$ROOT/donor_rows
TARGET_ROOT=$ROOT/target_rows
CAPTURE_ROOT=$ROOT/removal_captures
REMOVAL_VIDEO_ROOT=$ROOT/removal_run
OVERLAY_ROOT=$ROOT/removal_overlays
BASELINE10=/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_10step_case001460/seeds/seed_047326/probe_top100/videos/lora/cases/$CASE/original.mp4
BASELINE40=/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds/seed_047326/original.mp4
DONOR_SOURCE=/data/gaoya/agent-data/outputs/object_query_positive_delta_mask1x1_case001460/seed_047326/donor_rows
TARGET_SOURCE=/data/gaoya/agent-data/outputs/object_query_positive_delta_mask1x1_case001460/seed_047326/target_rows

mkdir -p "$ROOT/logs" "$CAPTURE_ROOT" "$OVERLAY_ROOT"
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

if [[ $(find "$DONOR_SOURCE" -maxdepth 1 -name 'step_*.npz' 2>/dev/null | wc -l) -eq 20 ]]; then
  [[ -e "$DONOR_ROOT" ]] || ln -s "$DONOR_SOURCE" "$DONOR_ROOT"
else
  mkdir -p "$DONOR_ROOT"
  DONOR_VIDEO_ROOT=$ROOT/donor_run
  link_original "$BASELINE10" "$DONOR_VIDEO_ROOT"
  "$PYTHON" "$WORKER" --mode donor --seed 47326 \
    --donor-root "$DONOR_ROOT" --mapping-csv "$MAPPING" --capture-root "$CAPTURE_ROOT" \
    --input-json-list "$CASE_LIST" --output-root "$DONOR_VIDEO_ROOT"
fi

if [[ $(find "$TARGET_SOURCE" -maxdepth 1 -name 'step_*.npz' 2>/dev/null | wc -l) -eq 80 ]]; then
  [[ -e "$TARGET_ROOT" ]] || ln -s "$TARGET_SOURCE" "$TARGET_ROOT"
else
  mkdir -p "$TARGET_ROOT"
  TARGET_VIDEO_ROOT=$ROOT/target_run
  link_original "$BASELINE40" "$TARGET_VIDEO_ROOT"
  "$PYTHON" "$WORKER" --mode target --seed 47326 \
    --donor-root "$TARGET_ROOT" --mapping-csv "$MAPPING" --capture-root "$CAPTURE_ROOT" \
    --input-json-list "$CASE_LIST" --output-root "$TARGET_VIDEO_ROOT"
fi

link_original "$BASELINE40" "$REMOVAL_VIDEO_ROOT"
"$PYTHON" "$WORKER" --mode similarity_delta_mask_removal --seed 47326 \
  --donor-root "$DONOR_ROOT" --mapping-csv "$MAPPING" --capture-root "$CAPTURE_ROOT" \
  --input-json-list "$CASE_LIST" --output-root "$REMOVAL_VIDEO_ROOT"

VIDEO=$REMOVAL_VIDEO_ROOT/lora/cases/$CASE/top100_steps_00_40.mp4
"$PYTHON" "$RENDERER" --capture-root "$CAPTURE_ROOT" --video "$VIDEO" --output-root "$OVERLAY_ROOT"

printf 'seed=47326\ngpu=%s\nmatching=per_head_similarity_csv\nthreshold=P95_positive_delta_per_query_per_latent_frame\nmask_kernel=1x1_no_expansion\napply_steps=S000-S039\ncompleted=%s\n' \
  "$GPU" "$(date -u +%FT%TZ)" > "$ROOT/complete"
