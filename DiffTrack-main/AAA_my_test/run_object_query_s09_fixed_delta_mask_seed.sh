#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?seed is required}"
GPU="${2:?physical GPU index is required}"
printf -v SEED_PAD '%06d' "$SEED"

BASE_ROOT=/data/gaoya/agent-data/outputs/object_query_s09_fixed_delta_mask_multiseed_case001460/seed_$SEED_PAD
SHARED_ROOT=$BASE_ROOT/shared
DIFFTRACK=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER=$DIFFTRACK/AAA_my_test/run_object_query_reverse_attention_transplant_worker.py
RENDERER=$DIFFTRACK/AAA_my_test/render_positive_delta_mask_top100_mean.py
CASE=0613pybullet_sample_001460_w002
CASE_JSON=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/$CASE.json
CASE_LIST=$BASE_ROOT/case_list.txt
MAPPING=/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40/analysis/reverse/per_head_best_matches_by_sample_reverse.csv
REGION_CACHE=/data/gaoya/agent-data/cache/test100_51_grounded_sam2_regions/case_test100_51_048_0613pybullet_sample_001460_w002
DONOR_ROOT=$SHARED_ROOT/donor_rows
TARGET_ROOT=$SHARED_ROOT/target_rows
DONOR_VIDEO_ROOT=$SHARED_ROOT/donor_run
TARGET_VIDEO_ROOT=$SHARED_ROOT/target_run
BASELINE10=/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_10step_case001460/seeds/seed_$SEED_PAD/probe_top100/videos/lora/cases/$CASE/original.mp4
BASELINE40=/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds/seed_$SEED_PAD/original.mp4

mkdir -p "$BASE_ROOT/logs" "$DONOR_ROOT" "$TARGET_ROOT"
printf '%s\n' "$CASE_JSON" > "$CASE_LIST"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OBJECT_QUERY_REGION_CACHE="$REGION_CACHE"
export QK_ATTENTION_CAPTURE_CASE="$CASE"
cd "$DIFFTRACK"

link_original() {
  local source="$1" run_root="$2"
  local target="$run_root/lora/cases/$CASE/original.mp4"
  [[ -s "$source" ]] || { echo "missing baseline: $source" >&2; exit 4; }
  mkdir -p "$(dirname "$target")"
  [[ -e "$target" ]] || ln -s "$source" "$target"
}

if [[ $(find "$DONOR_ROOT" -maxdepth 1 -name 'step_*.npz' | wc -l) -ne 20 ]]; then
  link_original "$BASELINE10" "$DONOR_VIDEO_ROOT"
  "$PYTHON" "$WORKER" --mode donor --seed "$SEED" \
    --donor-root "$DONOR_ROOT" --mapping-csv "$MAPPING" --capture-root "$SHARED_ROOT/donor_captures" \
    --input-json-list "$CASE_LIST" --output-root "$DONOR_VIDEO_ROOT"
fi

if [[ $(find "$TARGET_ROOT" -maxdepth 1 -name 'step_*.npz' | wc -l) -ne 80 ]]; then
  link_original "$BASELINE40" "$TARGET_VIDEO_ROOT"
  "$PYTHON" "$WORKER" --mode target --seed "$SEED" \
    --donor-root "$TARGET_ROOT" --mapping-csv "$MAPPING" --capture-root "$SHARED_ROOT/target_captures" \
    --input-json-list "$CASE_LIST" --output-root "$TARGET_VIDEO_ROOT"
fi

for KERNEL in 1 2 3; do
  ROOT=$BASE_ROOT/mask_${KERNEL}x${KERNEL}
  CAPTURE_ROOT=$ROOT/removal_captures
  VIDEO_ROOT=$ROOT/removal_run
  OVERLAY_ROOT=$ROOT/removal_overlays
  mkdir -p "$ROOT/logs" "$CAPTURE_ROOT" "$OVERLAY_ROOT"
  [[ -e "$ROOT/donor_rows" ]] || ln -s "$DONOR_ROOT" "$ROOT/donor_rows"
  [[ -e "$ROOT/target_rows" ]] || ln -s "$TARGET_ROOT" "$ROOT/target_rows"
  link_original "$BASELINE40" "$VIDEO_ROOT"
  VIDEO=$VIDEO_ROOT/lora/cases/$CASE/top100_steps_00_40.mp4
  if [[ ! -s "$VIDEO" ]]; then
    "$PYTHON" "$WORKER" --mode s09_fixed_delta_mask_removal --seed "$SEED" --mask-kernel "$KERNEL" \
      --donor-root "$ROOT/donor_rows" --mapping-csv "$MAPPING" --capture-root "$CAPTURE_ROOT" \
      --input-json-list "$CASE_LIST" --output-root "$VIDEO_ROOT"
  fi
  "$PYTHON" "$RENDERER" --capture-root "$CAPTURE_ROOT" --video "$VIDEO" --output-root "$OVERLAY_ROOT"
  printf 'seed=%s\ngpu=%s\nmask_source=A40_S09_minus_A10_S09_positive_P95\nmask_kernel=%sx%s\nrenormalize=true\napply_steps=S000-S009\ncompleted=%s\n' \
    "$SEED" "$GPU" "$KERNEL" "$KERNEL" "$(date -u +%FT%TZ)" > "$ROOT/complete"
done

printf 'seed=%s\ngpu=%s\ncompleted=%s\n' "$SEED" "$GPU" "$(date -u +%FT%TZ)" > "$BASE_ROOT/complete"
