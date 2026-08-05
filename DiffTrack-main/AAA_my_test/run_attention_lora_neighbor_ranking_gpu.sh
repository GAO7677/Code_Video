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
ROOT="${ATTENTION_NEIGHBOR_RANKING_ROOT:-/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed_sweep_case001460}"
CURRENT="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460"
SEEDS_FILE="${ROOT}/seeds.txt"
CASE_LIST="/data/gaoya/agent-data/outputs/attention_probability_mono_scale_steps40_frames49_case001460/case_list.txt"
CASE_KEY="0613pybullet_sample_001460_w002"
CRITERIA=(strict_score allblock_purity allblock_min_purity balanced uniformity joint mass pck32)
if [[ -n "${ATTENTION_NEIGHBOR_PROFILES:-}" ]]; then
  read -r -a PROFILES <<< "${ATTENTION_NEIGHBOR_PROFILES}"
else
  PROFILES=(alpha090 alpha150 zero uniform temporal_causal strict_past strict_future head_output_zero)
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${ROOT}/logs"
FIXED_SEED="${ATTENTION_RANKING_FIXED_SEED:-}"
if [[ -n "${FIXED_SEED}" ]]; then
  SEEDS=("${FIXED_SEED}")
else
  mapfile -t SEEDS < "${SEEDS_FILE}"
fi

link_original() {
  local seed="$1" run_root="$2"
  local source="${CURRENT}/seeds/seed_$(printf '%06d' "${seed}")/original.mp4"
  local target="${run_root}/videos/lora/cases/${CASE_KEY}/original.mp4"
  [[ -s "${source}" ]] || { echo "Missing original: ${source}" >&2; return 1; }
  mkdir -p "$(dirname "${target}")"
  if [[ ! -e "${target}" ]]; then
    ln "${source}" "${target}" 2>/dev/null || cp --reflink=auto "${source}" "${target}"
  fi
}

run_profile() {
  local seed="$1" criterion="$2" stage="$3" profile="$4"
  local seed_root="${ROOT}/seeds/seed_$(printf '%06d' "${seed}")"
  local run_root="${seed_root}/${criterion}/${stage}/${profile}"
  local complete="${run_root}/complete"
  [[ -f "${complete}" ]] && return
  mkdir -p "${run_root}/heatmaps" "${run_root}/videos"
  link_original "${seed}" "${run_root}"
  local mode alpha capture_step=39
  [[ "${stage}" == "steps00_09" ]] && capture_step=9
  case "${profile}" in
    alpha090) mode=probability_mono_scale; alpha=0.9 ;;
    alpha150) mode=probability_mono_scale; alpha=1.5 ;;
    zero) mode=probability_zero; alpha=0 ;;
    uniform) mode=probability_uniform; alpha=0 ;;
    temporal_causal) mode=probability_temporal_causal; alpha=0 ;;
    strict_past) mode=probability_strict_past; alpha=0 ;;
    strict_future) mode=probability_strict_future; alpha=0 ;;
    head_output_zero) mode=head_output_zero; alpha=0 ;;
    identity) mode=probability_identity; alpha=0 ;;
    *) echo "Unknown profile ${profile}" >&2; exit 2 ;;
  esac
  cd "${DIFFTRACK}"
  ATTENTION_NOISE_MODE="${mode}" \
  ATTENTION_NOISE_ALPHA="${alpha}" \
  ATTENTION_NOISE_SEED="${seed}" \
  QK_ATTENTION_NOISE_SEED="${seed}" \
  ATTENTION_MASK_LATENT_FRAMES=13 \
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
      --ranking-criterion "${criterion}" \
      --input-json-list "${CASE_LIST}" \
      --output-root "${run_root}/videos"
  printf 'seed=%s\ngpu=%s\ncriterion=%s\nstage=%s\nprofile=%s\ncompleted=%s\n' \
    "${seed}" "${GPU}" "${criterion}" "${stage}" "${profile}" \
    "$(date -u +%FT%TZ)" > "${complete}"
}

if [[ -n "${FIXED_SEED}" ]]; then
  task_index=0
  seed="${FIXED_SEED}"
  for criterion in "${CRITERIA[@]}"; do
    for stage in all_steps steps00_09; do
      for profile in "${PROFILES[@]}"; do
        if (( task_index % NUM_GPUS == GPU )); then
          run_profile "${seed}" "${criterion}" "${stage}" "${profile}"
        fi
        task_index=$((task_index + 1))
      done
    done
  done
else
  for index in "${!SEEDS[@]}"; do
    (( index % NUM_GPUS == GPU )) || continue
    seed="${SEEDS[index]}"
    for criterion in "${CRITERIA[@]}"; do
      for stage in all_steps steps00_09; do
        for profile in "${PROFILES[@]}"; do
          run_profile "${seed}" "${criterion}" "${stage}" "${profile}"
        done
      done
    done
  done
fi

COMPLETE_NAME="${ATTENTION_NEIGHBOR_COMPLETE_NAME:-gpu${GPU}.complete}"
printf 'gpu=%s\ncompleted=%s\n' "${GPU}" "$(date -u +%FT%TZ)" > "${ROOT}/logs/${COMPLETE_NAME}"
