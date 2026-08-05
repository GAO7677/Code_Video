#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU SEED}"
SEED="${2:?usage: $0 GPU SEED}"
DIFFTRACK=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER="$DIFFTRACK/AAA_my_test/run_attention_lora_seed_sweep_worker.py"
ROOT=/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40_baseline_official_ti2v
CASE=0613pybullet_sample_001460_w002
SID=$(printf '%06d' "$SEED")
CASE_LIST="$ROOT/case_list.txt"
REGION_CACHE=/data/gaoya/agent-data/cache/test100_51_grounded_sam2_regions/case_test100_51_048_0613pybullet_sample_001460_w002

mkdir -p "$ROOT/logs" "$ROOT/seeds/seed_$SID"
printf '%s\n' "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/$CASE.json" > "$CASE_LIST"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$DIFFTRACK"

run_schedule() {
  local steps="$1" expected="$2"
  local run_root="$ROOT/seeds/seed_$SID/steps$steps"
  local capture_root="$run_root/captures"
  local video_root="$run_root/videos"
  local count
  mkdir -p "$capture_root" "$video_root"
  count=$(find "$capture_root" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  [[ "$count" -ge "$expected" && -f "$run_root/complete" ]] && return
  find "$capture_root" -maxdepth 1 -name '*.npz' -delete
  rm -f "$run_root/complete"
  ATTENTION_NUM_INFERENCE_STEPS="$steps" \
  ATTENTION_NOISE_MODE=probability_object_query_trajectory_probe \
  ATTENTION_NOISE_ALPHA=0 ATTENTION_NOISE_SEED="$SEED" QK_ATTENTION_NOISE_SEED="$SEED" \
  ATTENTION_EXTREME_COUNT=100 ATTENTION_GROUP_FILTER=top ATTENTION_CFG_BRANCH_MODE=both \
  ATTENTION_RANKING_MODEL=baseline \
  WAN_BASELINE_PIPELINE=official_ti2v \
  ATTENTION_MASK_LATENT_FRAMES=13 ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
  OBJECT_GROUP_ACTIVE_STEP_END=$((steps - 1)) OBJECT_GROUP_EXPECTED_HEADS=100 \
  OBJECT_QUERY_REGION_CACHE="$REGION_CACHE" \
  OBJECT_STEP_ALIGNMENT_CAPTURE_ROOT="$capture_root" \
  QK_ATTENTION_CAPTURE_MODEL=baseline QK_ATTENTION_CAPTURE_CASE="$CASE" \
    "$PYTHON" "$WORKER" --seed "$SEED" --model baseline \
      --profile object_query_main_component --stage all_steps --ranking-criterion pck32 \
      --input-json-list "$CASE_LIST" --output-root "$video_root"
  count=$(find "$capture_root" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  [[ "$count" -ge "$expected" ]] || {
    echo "Incomplete baseline steps${steps} batched capture: $count/$expected" >&2
    exit 1
  }
  printf 'gpu=%s\nseed=%s\nmodel=baseline\npipeline=official_wan22_ti2v_5b\nconditioning=input_image_only\ninference_steps=%s\nheads=100\nobjects=2\nfiles=%s\ncompleted=%s\n' \
    "$GPU" "$SEED" "$steps" "$count" "$(date -u +%FT%TZ)" > "$run_root/complete"
}

rm -f "$ROOT/seeds/seed_$SID/complete"
run_schedule 40 80
run_schedule 10 20
printf 'gpu=%s\nseed=%s\nmodel=baseline\npipeline=official_wan22_ti2v_5b\nconditioning=input_image_only\ncompleted=%s\n' \
  "$GPU" "$SEED" "$(date -u +%FT%TZ)" > "$ROOT/seeds/seed_$SID/complete"
