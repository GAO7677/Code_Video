#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_ID NUM_GPUS" >&2
  exit 2
fi

GPU="$1"
NUM_GPUS="$2"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${DIFFTRACK}/AAA_my_test/run_attention_lora_seed_sweep_worker.py"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460"
SEEDS_FILE="${ROOT}/seeds.txt"
CASE_LIST="/data/gaoya/agent-data/outputs/attention_probability_mono_scale_steps40_frames49_case001460/case_list.txt"
CASE_KEY="0613pybullet_sample_001460_w002"

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${ROOT}/logs"
mapfile -t SEEDS < "${SEEDS_FILE}"

link_original() {
  local canonical="$1"
  local target_root="$2"
  local target="${target_root}/videos/lora/cases/${CASE_KEY}/original.mp4"
  mkdir -p "$(dirname "${target}")"
  if [[ ! -s "${target}" ]]; then
    ln "${canonical}" "${target}" 2>/dev/null || cp --reflink=auto "${canonical}" "${target}"
  fi
}

run_profile() {
  local seed="$1"
  local stage="$2"
  local profile="$3"
  local seed_root="${ROOT}/seeds/seed_$(printf '%06d' "${seed}")"
  local run_root="${seed_root}/${stage}/${profile}"
  local complete="${run_root}/complete"
  local capture_step=39
  [[ "${stage}" == "steps00_09" ]] && capture_step=9
  if [[ -f "${complete}" ]]; then
    return
  fi
  mkdir -p "${run_root}/heatmaps" "${run_root}/videos"
  if [[ "${profile}" != "alpha090" || "${stage}" != "all_steps" ]]; then
    link_original "${seed_root}/original.mp4" "${run_root}"
  fi
  local mode alpha
  case "${profile}" in
    alpha090) mode=probability_mono_scale; alpha=0.9 ;;
    alpha150) mode=probability_mono_scale; alpha=1.5 ;;
    zero) mode=probability_zero; alpha=0 ;;
    uniform) mode=probability_uniform; alpha=0 ;;
    temporal_causal) mode=probability_temporal_causal; alpha=0 ;;
    head_output_zero) mode=head_output_zero; alpha=0 ;;
    *) echo "Unknown profile ${profile}" >&2; exit 2 ;;
  esac
  cd "${DIFFTRACK}"
  ATTENTION_NOISE_MODE="${mode}" \
  ATTENTION_NOISE_ALPHA="${alpha}" \
  ATTENTION_NOISE_SEED="${seed}" \
  QK_ATTENTION_NOISE_SEED="${seed}" \
  ATTENTION_MASK_LATENT_FRAMES=7 \
  QK_ATTENTION_CAPTURE_ROOT="${run_root}/heatmaps" \
  QK_ATTENTION_CAPTURE_STEP="${capture_step}" \
  QK_ATTENTION_CAPTURE_MODEL=lora \
  QK_ATTENTION_CAPTURE_CASE="${CASE_KEY}" \
  QK_ATTENTION_CAPTURE_PER_HEAD=0 \
  QK_ATTENTION_CAPTURE_SMALL_SIZE=416 \
  QK_ATTENTION_CAPTURE_LATENT_FRAMES=13 \
  "${PYTHON}" "${WORKER}" \
    --seed "${seed}" \
    --profile "${profile}" \
    --stage "${stage}" \
    --input-json-list "${CASE_LIST}" \
    --output-root "${run_root}/videos"
  if [[ "${profile}" == "alpha090" && "${stage}" == "all_steps" ]]; then
    canonical="${run_root}/videos/lora/cases/${CASE_KEY}/original.mp4"
    ln "${canonical}" "${seed_root}/original.mp4" 2>/dev/null || cp --reflink=auto "${canonical}" "${seed_root}/original.mp4"
  fi
  printf 'seed=%s\ngpu=%s\nstage=%s\nprofile=%s\ncompleted=%s\n' \
    "${seed}" "${GPU}" "${stage}" "${profile}" "$(date -u +%FT%TZ)" > "${complete}"
}

for index in "${!SEEDS[@]}"; do
  if (( index % NUM_GPUS != GPU )); then
    continue
  fi
  seed="${SEEDS[index]}"
  run_profile "${seed}" all_steps alpha090
  for stage in all_steps steps00_09; do
    for profile in alpha090 alpha150 zero uniform temporal_causal head_output_zero; do
      [[ "${stage}" == all_steps && "${profile}" == alpha090 ]] && continue
      run_profile "${seed}" "${stage}" "${profile}"
    done
  done
done

printf 'gpu=%s\ncompleted=%s\n' "${GPU}" "$(date -u +%FT%TZ)" > "${ROOT}/logs/gpu${GPU}.complete"
