#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 GPU_ID SHARD_INDEX NUM_SHARDS" >&2
  exit 2
fi

GPU="$1"
SHARD_INDEX="$2"
NUM_SHARDS="$3"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${DIFFTRACK}/AAA_my_test/run_attention_lora_seed_sweep_worker.py"
SOURCE_LIST="/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_pck32_temporal_test5_seed000851"
BASELINE="/data/gaoya/agent-data/outputs/attention_probability_noise_unified_steps40_frames49_test5/lora/alpha090_count100/videos/lora/cases"
UNIQUE_LIST="${ROOT}/test_5_unique.txt"
SHARD_LIST="${ROOT}/case_lists/shard_${SHARD_INDEX}_of_${NUM_SHARDS}.txt"

mkdir -p "${ROOT}/logs" "${ROOT}/case_lists"
awk 'NF && !seen[$0]++' "${SOURCE_LIST}" > "${UNIQUE_LIST}"
awk -v shard="${SHARD_INDEX}" -v total="${NUM_SHARDS}" \
  '((NR - 1) % total) == shard' "${UNIQUE_LIST}" > "${SHARD_LIST}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for stage in steps00_09 all_steps; do
  run_root="${ROOT}/${stage}/temporal_causal"
  complete="${run_root}/gpu${GPU}_shard${SHARD_INDEX}of${NUM_SHARDS}.complete"
  [[ -f "${complete}" ]] && continue
  mkdir -p "${run_root}/videos/lora/cases"
  while IFS= read -r input_json; do
    [[ -n "${input_json}" ]] || continue
    case_key="$(basename "${input_json}" .json)"
    source="${BASELINE}/${case_key}/original.mp4"
    target="${run_root}/videos/lora/cases/${case_key}/original.mp4"
    [[ -s "${source}" ]] || { echo "Missing Wan+LoRA Original: ${source}" >&2; exit 1; }
    mkdir -p "$(dirname "${target}")"
    if [[ ! -e "${target}" ]]; then
      ln "${source}" "${target}" 2>/dev/null || cp --reflink=auto "${source}" "${target}"
    fi
  done < "${SHARD_LIST}"

  cd "${DIFFTRACK}"
  ATTENTION_NOISE_MODE=probability_temporal_causal \
  ATTENTION_NOISE_ALPHA=0 \
  ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_NOISE_SEED=851 \
  ATTENTION_GROUP_FILTER=top \
  ATTENTION_CFG_BRANCH_MODE=both \
  ATTENTION_MASK_LATENT_FRAMES=13 \
  ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
    "${PYTHON}" "${WORKER}" \
      --seed 851 \
      --profile temporal_causal \
      --stage "${stage}" \
      --ranking-criterion pck32 \
      --input-json-list "${SHARD_LIST}" \
      --output-root "${run_root}/videos"

  printf 'gpu=%s\nshard=%s/%s\nseed=851\nsteps=40\nframes=49\nprofile=temporal_causal\ncriterion=lora_pck32\ngroup=top100\nstage=%s\ncompleted=%s\n' \
    "${GPU}" "${SHARD_INDEX}" "${NUM_SHARDS}" "${stage}" "$(date -u +%FT%TZ)" > "${complete}"
done

