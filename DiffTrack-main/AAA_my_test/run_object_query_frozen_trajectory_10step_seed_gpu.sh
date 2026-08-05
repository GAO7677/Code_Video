#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU SEED}"
SEED="${2:?usage: $0 GPU SEED}"
DIFFTRACK=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER="$DIFFTRACK/AAA_my_test/run_attention_lora_seed_sweep_worker.py"
ROOT=/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_10step_case001460
CASE=0613pybullet_sample_001460_w002
SID=$(printf '%06d' "$SEED")
SEED_ROOT="$ROOT/seeds/seed_$SID"
CASE_LIST="$ROOT/case_list.txt"
PROBE100_ROOT="$SEED_ROOT/probe_top100/captures"
PROBE100_VIDEO_ROOT="$SEED_ROOT/probe_top100/videos"
PROBE30_ROOT="$SEED_ROOT/probe_top30/captures"
PROBE30_VIDEO_ROOT="$SEED_ROOT/probe_top30/videos"
BASELINE="$PROBE100_VIDEO_ROOT/lora/cases/$CASE/original.mp4"

mkdir -p "$ROOT/logs" "$SEED_ROOT"
printf '%s\n' "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/$CASE.json" > "$CASE_LIST"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$DIFFTRACK"

link_baseline() {
  local video_root="$1"
  local target="$video_root/lora/cases/$CASE/original.mp4"
  mkdir -p "$(dirname "$target")"
  [[ -e "$target" ]] || ln "$BASELINE" "$target" 2>/dev/null || cp --reflink=auto "$BASELINE" "$target"
}

run_probe() {
  local heads="$1" capture_root="$2" video_root="$3" marker="$4"
  local count
  mkdir -p "$capture_root"
  count=$(find "$capture_root" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  if [[ "$count" -ge 40 ]]; then
    if [[ ! -f "$marker" ]]; then
      mkdir -p "$(dirname "$marker")"
      printf 'gpu=%s\nseed=%s\ninference_steps=10\nmean_heads=%s\nrecovered_from_captures=40\ncompleted=%s\n' \
        "$GPU" "$SEED" "$heads" "$(date -u +%FT%TZ)" > "$marker"
    fi
    return
  fi
  [[ "$heads" == 30 ]] && link_baseline "$video_root"
  ATTENTION_NUM_INFERENCE_STEPS=10 \
  ATTENTION_NOISE_MODE=probability_object_query_trajectory_probe \
  ATTENTION_NOISE_ALPHA=0 ATTENTION_NOISE_SEED="$SEED" QK_ATTENTION_NOISE_SEED="$SEED" \
  ATTENTION_EXTREME_COUNT="$heads" ATTENTION_GROUP_FILTER=top ATTENTION_CFG_BRANCH_MODE=both \
  ATTENTION_MASK_LATENT_FRAMES=13 ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
  OBJECT_GROUP_ACTIVE_STEP_END=9 OBJECT_GROUP_EXPECTED_HEADS="$heads" \
  OBJECT_CONTINUITY_HIGH_QUANTILE=0.90 OBJECT_CONTINUITY_NEIGHBOR_RADIUS=2 \
  OBJECT_CONTINUITY_MAIN_COMPONENT_TOPK=5 OBJECT_TRAJECTORY_PROBE_ROOT="$capture_root" \
    "$PYTHON" "$WORKER" --seed "$SEED" \
      --profile object_query_main_component --stage all_steps --ranking-criterion pck32 \
      --input-json-list "$CASE_LIST" --output-root "$video_root"
  count=$(find "$capture_root" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  [[ "$count" -ge 40 ]] || { echo "Incomplete Top${heads} 10-step probe: $count/40" >&2; exit 1; }
  mkdir -p "$(dirname "$marker")"
  printf 'gpu=%s\nseed=%s\ninference_steps=10\nmean_heads=%s\ncompleted=%s\n' \
    "$GPU" "$SEED" "$heads" "$(date -u +%FT%TZ)" > "$marker"
}

run_probe 100 "$PROBE100_ROOT" "$PROBE100_VIDEO_ROOT" "$SEED_ROOT/probe_top100/complete"
[[ -s "$BASELINE" ]] || { echo "Missing 10-step baseline: $BASELINE" >&2; exit 1; }
if [[ "${TENSTEP_ORIGINAL_ONLY:-0}" == 1 ]]; then
  COMMON_ROOT="$SEED_ROOT/common_top100/overlays"
  "$PYTHON" AAA_my_test/render_object_query_probe_mean.py \
    --probe-root "$PROBE100_ROOT" --render-root "$COMMON_ROOT" --video "$BASELINE"
  printf 'gpu=%s\nseed=%s\ninference_steps=10\nmean_heads=100\nmode=original_no_intervention\ncompleted=%s\n' \
    "$GPU" "$SEED" "$(date -u +%FT%TZ)" > "$SEED_ROOT/original_top100_complete"
  exit 0
fi
run_probe 30 "$PROBE30_ROOT" "$PROBE30_VIDEO_ROOT" "$SEED_ROOT/probe_top30/complete"

build_variant() {
  local label="$1" quantile="$2" probe_root="$3"
  shift 3
  local mask_root="$SEED_ROOT/trajectory/$label/masks"
  local render_root="$SEED_ROOT/trajectory/$label/overlays"
  local count
  mkdir -p "$mask_root" "$render_root"
  count=$(find "$mask_root" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  if [[ "$count" -lt 40 || ! -f "$render_root/manifest.json" ]]; then
    "$PYTHON" AAA_my_test/build_object_query_frozen_trajectory_masks.py \
      --probe-root "$probe_root" --output-root "$mask_root" \
      --render-root "$render_root" --video "$BASELINE" \
      --quantile "$quantile" --radius 2 "$@"
  fi
}

build_variant p95 0.95 "$PROBE100_ROOT"
build_variant p99 0.99 "$PROBE100_ROOT"
build_variant p95_single 0.95 "$PROBE100_ROOT" --single-component
build_variant p99_single 0.99 "$PROBE100_ROOT" --single-component
build_variant p95_single_d1 0.95 "$PROBE100_ROOT" --single-component --removal-dilate-radius 1
build_variant p99_single_d1 0.99 "$PROBE100_ROOT" --single-component --removal-dilate-radius 1
build_variant p95_single_bt3_d1 0.95 "$PROBE30_ROOT" --single-component --removal-dilate-radius 1 --backtrack-frames 3
build_variant p99_single_bt3_d1 0.99 "$PROBE30_ROOT" --single-component --removal-dilate-radius 1 --backtrack-frames 3

run_apply() {
  local label="$1"
  local run_root="$SEED_ROOT/apply/$label"
  local video_root="$run_root/videos"
  local capture_root="$run_root/captures"
  local overlay_root="$run_root/overlays"
  local mask_root="$SEED_ROOT/trajectory/$label/masks"
  local video
  [[ -f "$run_root/complete" ]] && return
  mkdir -p "$capture_root" "$overlay_root"
  link_baseline "$video_root"
  ATTENTION_NUM_INFERENCE_STEPS=10 \
  ATTENTION_NOISE_MODE=probability_object_query_frozen_trajectory \
  ATTENTION_NOISE_ALPHA=0 ATTENTION_NOISE_SEED="$SEED" QK_ATTENTION_NOISE_SEED="$SEED" \
  ATTENTION_EXTREME_COUNT=100 ATTENTION_GROUP_FILTER=top ATTENTION_CFG_BRANCH_MODE=both \
  ATTENTION_MASK_LATENT_FRAMES=13 ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
  OBJECT_GROUP_ACTIVE_STEP_END=9 OBJECT_GROUP_EXPECTED_HEADS=100 \
  OBJECT_TRAJECTORY_MASK_ROOT="$mask_root" \
  OBJECT_TRAJECTORY_APPLY_CAPTURE_ROOT="$capture_root" \
    "$PYTHON" "$WORKER" --seed "$SEED" \
      --profile object_query_main_component --stage all_steps --ranking-criterion pck32 \
      --input-json-list "$CASE_LIST" --output-root "$video_root"
  video="$video_root/lora/cases/$CASE/top100_steps_00_40.mp4"
  [[ -s "$video" ]] || { echo "Missing 10-step intervention video: $video" >&2; exit 1; }
  "$PYTHON" AAA_my_test/render_object_query_frozen_trajectory_apply.py \
    --capture-root "$capture_root" --video "$video" --output-root "$overlay_root"
  printf 'gpu=%s\nseed=%s\ninference_steps=10\nmask=%s\nmask_mean_heads=%s\napply_heads=100\ncompleted=%s\n' \
    "$GPU" "$SEED" "$label" "$([[ "$label" == *bt3* ]] && echo 30 || echo 100)" \
    "$(date -u +%FT%TZ)" > "$run_root/complete"
}

for label in p95 p99 p95_single p99_single p95_single_d1 p99_single_d1 p95_single_bt3_d1 p99_single_bt3_d1; do
  run_apply "$label"
done

printf 'gpu=%s\nseed=%s\ninference_steps=10\ncompleted=%s\n' \
  "$GPU" "$SEED" "$(date -u +%FT%TZ)" > "$SEED_ROOT/complete"
