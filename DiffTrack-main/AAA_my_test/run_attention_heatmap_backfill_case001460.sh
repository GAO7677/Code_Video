#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_ID {mono0|mono1|mono2|mono3|replacement_baseline|replacement_lora}" >&2
  exit 2
fi

GPU="$1"
PROFILE="$2"
CASE_KEY="0613pybullet_sample_001460_w002"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
MONO_RUNNER="${DIFFTRACK}/AAA_my_test/run_attention_qk_mono_scale_unified40_49f_gpu_queue.sh"
REPLACEMENT_WORKER="${DIFFTRACK}/AAA_my_test/run_pck_step_adaptive_attention_replacement_49f_worker.py"
MONO_ROOT="/data/gaoya/agent-data/outputs/attention_probability_mono_scale_steps40_frames49_case001460"
REPLACEMENT_ROOT="/data/gaoya/agent-data/outputs/attention_probability_replacement_steps40_frames49_test5"
CASE_LIST="${MONO_ROOT}/case_list.txt"
LOG_ROOT="${MONO_ROOT}/logs/backfill"

if [[ "${GPU}" == "4" || "${GPU}" == "6" || "${GPU}" == "7" ]]; then
  echo "GPU ${GPU} is reserved and will not be used." >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"
if [[ ! -s "${CASE_LIST}" ]]; then
  echo "Missing single-case list: ${CASE_LIST}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

run_mono_profile() {
  local queue_profile="$1"
  ATTENTION_MONO_TEST_LIST="${CASE_LIST}" \
  ATTENTION_MONO_OUTPUT_ROOT="${MONO_ROOT}" \
  ATTENTION_MONO_CAPTURE_CASE="${CASE_KEY}" \
  ATTENTION_MONO_CAPTURE_PER_HEAD=0 \
  MONO_SCALE_PROFILE="${queue_profile}" \
  bash "${MONO_RUNNER}" "${GPU}"
}

run_replacement() {
  local model="$1"
  local intervention="$2"
  local run_root="${REPLACEMENT_ROOT}/${model}/${intervention}_count100"
  local capture_video_root="${run_root}/_heatmap_backfill_videos"
  mkdir -p "${run_root}/heatmaps" "${capture_video_root}"
  cd "${DIFFTRACK}"
  ATTENTION_NOISE_MODE="probability_${intervention}" \
  ATTENTION_NOISE_ALPHA=0 \
  ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_CAPTURE_ROOT="${run_root}/heatmaps" \
  QK_ATTENTION_CAPTURE_STEP=39 \
  QK_ATTENTION_CAPTURE_MODEL="${model}" \
  QK_ATTENTION_CAPTURE_CASE="${CASE_KEY}" \
  QK_ATTENTION_CAPTURE_PER_HEAD=0 \
  "${PYTHON}" "${REPLACEMENT_WORKER}" \
    --model "${model}" \
    --input-json-list "${CASE_LIST}" \
    --output-root "${capture_video_root}" \
    --shard-index 0 \
    --num-shards 1 \
    --ranking-pool all720 \
    --extreme-count 100
}

case "${PROFILE}" in
  mono0) run_mono_profile 0 ;;
  mono1) run_mono_profile 1 ;;
  mono2) run_mono_profile 2 ;;
  mono3) run_mono_profile 3 ;;
  replacement_baseline)
    run_replacement baseline zero
    run_replacement baseline uniform
    ;;
  replacement_lora)
    run_replacement lora zero
    run_replacement lora uniform
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    exit 2
    ;;
esac

printf 'gpu=%s\nprofile=%s\ncase=%s\ncompleted=%s\nsteps=40\nframes=49\n' \
  "${GPU}" "${PROFILE}" "${CASE_KEY}" "$(date -u +%FT%TZ)" \
  > "${LOG_ROOT}/${PROFILE}.complete"
echo "CASE001460_HEATMAP_BACKFILL_COMPLETE gpu=${GPU} profile=${PROFILE}"
