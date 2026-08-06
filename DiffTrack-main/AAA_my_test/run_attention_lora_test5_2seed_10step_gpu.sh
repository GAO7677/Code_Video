#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 GPU_ID SHARD_ID NUM_SHARDS" >&2
  exit 2
fi

GPU="$1"
SHARD_ID="$2"
NUM_SHARDS="$3"
HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${HERE}/run_attention_lora_seed_sweep_worker.py"
QUEUE="${HERE}/attention_lora_test5_20case_10seed_queue.tsv"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_test5_20case_2seed_steps10"
SEEDS=(90094 35075)
PROFILES=(alpha090 alpha150 zero uniform temporal_causal strict_past strict_future head_output_zero)
IDLE_LIMIT_MIB="${ATTENTION_10STEP_IDLE_LIMIT_MIB:-12000}"
IDLE_POLLS="${ATTENTION_10STEP_IDLE_POLLS:-2}"
IDLE_INTERVAL="${ATTENTION_10STEP_IDLE_INTERVAL:-20}"

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${ROOT}/logs"
exec > >(tee -a "${ROOT}/logs/gpu${GPU}_shard${SHARD_ID}.log") 2>&1

wait_for_gpu() {
  local stable=0 used
  while (( stable < IDLE_POLLS )); do
    used="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')"
    if [[ -n "${used}" ]] && (( used < IDLE_LIMIT_MIB )); then
      stable=$((stable + 1))
      echo "[$(date -Is)] GPU${GPU} available ${stable}/${IDLE_POLLS}: ${used} MiB"
    else
      stable=0
      echo "[$(date -Is)] GPU${GPU} busy: ${used:-unknown} MiB; waiting"
    fi
    sleep "${IDLE_INTERVAL}"
  done
}

link_original() {
  local canonical="$1" target_root="$2"
  local target="${target_root}/videos/lora/cases/${CASE_KEY}/original.mp4"
  mkdir -p "$(dirname "${target}")"
  if [[ ! -s "${target}" ]]; then
    ln "${canonical}" "${target}" 2>/dev/null || cp --reflink=auto "${canonical}" "${target}"
  fi
}

run_profile() {
  local seed="$1" profile="$2"
  local seed_root="${CASE_ROOT}/seeds/seed_$(printf '%06d' "${seed}")"
  local run_root="${seed_root}/all_steps/${profile}"
  local complete="${run_root}/complete"
  local top_video="${run_root}/videos/lora/cases/${CASE_KEY}/top100_steps_00_40.mp4"
  local bottom_video="${run_root}/videos/lora/cases/${CASE_KEY}/bottom100_steps_00_40.mp4"
  if [[ -f "${complete}" && -s "${top_video}" && -s "${bottom_video}" ]]; then
    return
  fi
  mkdir -p "${run_root}/heatmaps" "${run_root}/videos"
  if [[ "${profile}" != "alpha090" ]]; then
    link_original "${seed_root}/original.mp4" "${run_root}"
  fi
  local mode alpha
  case "${profile}" in
    alpha090) mode=probability_mono_scale; alpha=0.9 ;;
    alpha150) mode=probability_mono_scale; alpha=1.5 ;;
    zero) mode=probability_zero; alpha=0 ;;
    uniform) mode=probability_uniform; alpha=0 ;;
    temporal_causal) mode=probability_temporal_causal; alpha=0 ;;
    strict_past) mode=probability_strict_past; alpha=0 ;;
    strict_future) mode=probability_strict_future; alpha=0 ;;
    head_output_zero) mode=head_output_zero; alpha=0 ;;
    *) echo "Unknown profile ${profile}" >&2; exit 2 ;;
  esac
  cd "${DIFFTRACK}"
  ATTENTION_NUM_INFERENCE_STEPS=10 \
  ATTENTION_NOISE_MODE="${mode}" \
  ATTENTION_NOISE_ALPHA="${alpha}" \
  ATTENTION_NOISE_SEED="${seed}" \
  QK_ATTENTION_NOISE_SEED="${seed}" \
  ATTENTION_MASK_LATENT_FRAMES=13 \
  QK_ATTENTION_CAPTURE_ROOT="${run_root}/heatmaps" \
  QK_ATTENTION_CAPTURE_STEP=9 \
  QK_ATTENTION_CAPTURE_MODEL=lora \
  QK_ATTENTION_CAPTURE_CASE="${CASE_KEY}" \
  QK_ATTENTION_CAPTURE_PER_HEAD=0 \
  QK_ATTENTION_CAPTURE_SMALL_SIZE=416 \
  QK_ATTENTION_CAPTURE_LATENT_FRAMES=13 \
    "${PYTHON}" "${WORKER}" \
      --seed "${seed}" \
      --profile "${profile}" \
      --stage all_steps \
      --input-json-list "${CASE_ROOT}/case_list.txt" \
      --output-root "${run_root}/videos"
  if [[ "${profile}" == "alpha090" ]]; then
    local canonical="${run_root}/videos/lora/cases/${CASE_KEY}/original.mp4"
    if [[ ! -e "${seed_root}/original.mp4" ]]; then
      ln "${canonical}" "${seed_root}/original.mp4" 2>/dev/null || cp --reflink=auto "${canonical}" "${seed_root}/original.mp4"
    fi
  fi
  printf 'seed=%s\ngpu=%s\nsteps=10\nprofile=%s\ncompleted=%s\n' \
    "${seed}" "${GPU}" "${profile}" "$(date -u +%FT%TZ)" > "${complete}"
}

wait_for_gpu
case_index=0
while IFS=$'\t' read -r CASE_KEY input_json; do
  if (( case_index % NUM_SHARDS != SHARD_ID )); then
    case_index=$((case_index + 1))
    continue
  fi
  CASE_ROOT="${ROOT}/cases/${CASE_KEY}"
  mkdir -p "${CASE_ROOT}/seeds" "${CASE_ROOT}/logs"
  printf '%s\n' "${input_json}" > "${CASE_ROOT}/case_list.txt"
  printf '90094\n35075\n' > "${CASE_ROOT}/seeds.txt"
  echo "[$(date -Is)] GPU${GPU} start ${CASE_KEY}"
  for seed in "${SEEDS[@]}"; do
    for profile in "${PROFILES[@]}"; do
      run_profile "${seed}" "${profile}"
    done
  done
  printf 'gpu=%s\nsteps=10\ncompleted=%s\n' "${GPU}" "$(date -u +%FT%TZ)" > "${CASE_ROOT}/COMPLETE"
  echo "[$(date -Is)] GPU${GPU} complete ${CASE_KEY}"
  case_index=$((case_index + 1))
done < "${QUEUE}"
printf 'gpu=%s\nshard=%s\ncompleted=%s\n' "${GPU}" "${SHARD_ID}" "$(date -u +%FT%TZ)" > "${ROOT}/SHARD${SHARD_ID}_COMPLETE"

