#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GPU_ID PROFILE [PROFILE ...]" >&2
  exit 2
fi

GPU="$1"
shift
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${DIFFTRACK}/AAA_my_test/run_attention_lora_seed_sweep_worker.py"
RENDERER="${DIFFTRACK}/AAA_my_test/render_object_query_attention_headwise_pck.py"
ROOT="/data/gaoya/agent-data/outputs/object_query_attention_overlay_headwise_pck_case001460_seed090094"
CASE_LIST="/data/gaoya/agent-data/outputs/attention_probability_mono_scale_steps40_frames49_case001460/case_list.txt"
CASE_KEY="0613pybullet_sample_001460_w002"
REGION_CACHE="/data/gaoya/agent-data/cache/test100_51_grounded_sam2_regions/case_test100_51_048_0613pybullet_sample_001460_w002"
SEED=90094

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${ROOT}/logs"

for profile in "$@"; do
  case "${profile}" in
    alpha090) mode=probability_mono_scale; alpha=0.9 ;;
    alpha150) mode=probability_mono_scale; alpha=1.5 ;;
    zero) mode=probability_zero; alpha=0 ;;
    uniform) mode=probability_uniform; alpha=0 ;;
    temporal_causal) mode=probability_temporal_causal; alpha=0 ;;
    strict_past) mode=probability_strict_past; alpha=0 ;;
    strict_future) mode=probability_strict_future; alpha=0 ;;
    exclude_current) mode=probability_exclude_current; alpha=0 ;;
    context_only) mode=probability_context_only; alpha=0 ;;
    *) echo "Unsupported profile ${profile}" >&2; exit 2 ;;
  esac
  run_root="${ROOT}/all_steps/${profile}"
  capture_root="${run_root}/captures"
  video_root="${run_root}/videos"
  overlay_root="${run_root}/overlays"
  mkdir -p "${capture_root}" "${video_root}" "${overlay_root}"
  cd "${DIFFTRACK}"
  ATTENTION_NOISE_MODE="${mode}" \
  ATTENTION_NOISE_ALPHA="${alpha}" \
  ATTENTION_NOISE_SEED="${SEED}" \
  QK_ATTENTION_NOISE_SEED="${SEED}" \
  ATTENTION_MASK_LATENT_FRAMES=13 \
  ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
  QK_ATTENTION_CAPTURE_ROOT="${run_root}/heatmaps" \
  QK_ATTENTION_CAPTURE_STEP=39 \
  QK_ATTENTION_CAPTURE_MODEL=lora \
  QK_ATTENTION_CAPTURE_CASE="${CASE_KEY}" \
  QK_ATTENTION_CAPTURE_PER_HEAD=0 \
  QK_ATTENTION_CAPTURE_SMALL_SIZE=416 \
  QK_ATTENTION_CAPTURE_LATENT_FRAMES=13 \
  OBJECT_QUERY_CAPTURE_PROTOCOL=headwise_pck \
  OBJECT_QUERY_RANKING_SELECTION="/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed090094_case001460/seeds/seed_090094/pck32/all_steps/alpha090/videos/selection.json" \
  OBJECT_QUERY_REGION_CACHE="${REGION_CACHE}" \
  OBJECT_QUERY_CAPTURE_ROOT="${capture_root}" \
  "${PYTHON}" "${WORKER}" \
    --seed "${SEED}" \
    --profile "${profile}" \
    --stage all_steps \
    --input-json-list "${CASE_LIST}" \
    --output-root "${video_root}"
  "${PYTHON}" "${RENDERER}" \
    --capture-root "${capture_root}" \
    --video-root "${video_root}/lora/cases/${CASE_KEY}" \
    --output-root "${overlay_root}"
  touch "${overlay_root}/complete"
  printf 'gpu=%s\nprofile=%s\ncompleted=%s\n' \
    "${GPU}" "${profile}" "$(date -u +%FT%TZ)" > "${run_root}/complete"
done
