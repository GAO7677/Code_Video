#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_ID {alpha090|alpha150|zero|uniform|temporal_causal}" >&2
  exit 2
fi

GPU="$1"
PROFILE="$2"
CASE_KEY="0613pybullet_sample_001460_w002"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
ALPHA_WORKER="${DIFFTRACK}/AAA_my_test/run_pck_step_adaptive_qk_probability_noise_00_09_49f_worker.py"
REPLACEMENT_WORKER="${DIFFTRACK}/AAA_my_test/run_pck_step_adaptive_attention_replacement_00_09_49f_worker.py"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/attention_wan_lora_steps00_09_case001460"
CASE_LIST="/data/gaoya/agent-data/outputs/attention_probability_mono_scale_steps40_frames49_case001460/case_list.txt"
LOG_ROOT="${OUTPUT_ROOT}/logs"

if [[ "${GPU}" == "4" || "${GPU}" == "6" || "${GPU}" == "7" ]]; then
  echo "GPU ${GPU} is reserved and will not be used." >&2
  exit 2
fi
if [[ ! -s "${CASE_LIST}" ]]; then
  echo "Missing single-case list: ${CASE_LIST}" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

run_alpha() {
  local alpha="$1"
  local alpha_tag="$2"
  local run_root="${OUTPUT_ROOT}/lora/alpha${alpha_tag}_count100"
  mkdir -p "${run_root}/heatmaps" "${run_root}/videos"
  cd "${DIFFTRACK}"
  ATTENTION_NOISE_MODE=probability_mono_scale \
  ATTENTION_NOISE_ALPHA="${alpha}" \
  ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_CAPTURE_ROOT="${run_root}/heatmaps" \
  QK_ATTENTION_CAPTURE_STEP=9 \
  QK_ATTENTION_CAPTURE_MODEL=lora \
  QK_ATTENTION_CAPTURE_CASE="${CASE_KEY}" \
  QK_ATTENTION_CAPTURE_PER_HEAD=0 \
  QK_ATTENTION_CAPTURE_SMALL_SIZE=416 \
  QK_ATTENTION_CAPTURE_LATENT_FRAMES=13 \
  "${PYTHON}" "${ALPHA_WORKER}" \
    --model lora \
    --input-json-list "${CASE_LIST}" \
    --output-root "${run_root}/videos" \
    --shard-index 0 \
    --num-shards 1 \
    --ranking-pool all720 \
    --extreme-count 100
}

run_replacement() {
  local intervention="$1"
  local run_root="${OUTPUT_ROOT}/lora/${intervention}_count100"
  mkdir -p "${run_root}/heatmaps" "${run_root}/videos"
  cd "${DIFFTRACK}"
  ATTENTION_NOISE_MODE="probability_${intervention}" \
  ATTENTION_NOISE_ALPHA=0 \
  ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_NOISE_SEED=851 \
  ATTENTION_MASK_LATENT_FRAMES=7 \
  QK_ATTENTION_CAPTURE_ROOT="${run_root}/heatmaps" \
  QK_ATTENTION_CAPTURE_STEP=9 \
  QK_ATTENTION_CAPTURE_MODEL=lora \
  QK_ATTENTION_CAPTURE_CASE="${CASE_KEY}" \
  "${PYTHON}" "${REPLACEMENT_WORKER}" \
    --model lora \
    --input-json-list "${CASE_LIST}" \
    --output-root "${run_root}/videos" \
    --shard-index 0 \
    --num-shards 1 \
    --ranking-pool all720 \
    --extreme-count 100
}

case "${PROFILE}" in
  alpha090) run_alpha 0.9 090 ;;
  alpha150) run_alpha 1.5 150 ;;
  zero) run_replacement zero ;;
  uniform) run_replacement uniform ;;
  temporal_causal) run_replacement temporal_causal ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    exit 2
    ;;
esac

printf 'gpu=%s\nprofile=%s\ncase=%s\nactive_steps=00-09\ncompleted=%s\nsteps=40\nframes=49\n' \
  "${GPU}" "${PROFILE}" "${CASE_KEY}" "$(date -u +%FT%TZ)" \
  > "${LOG_ROOT}/${PROFILE}.complete"
echo "WAN_LORA_STEPS00_09_COMPLETE gpu=${GPU} profile=${PROFILE}"
