#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?Usage: $0 GPU_ID}"
if [[ "${GPU}" == "4" || "${GPU}" == "6" || "${GPU}" == "7" ]]; then
  echo "GPU ${GPU} is reserved and will not be used." >&2
  exit 2
fi

DIFFTRACK=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER="${DIFFTRACK}/AAA_my_test/run_pck_step_adaptive_attention_replacement_49f_worker.py"
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/attention_probability_replacement_steps40_frames49_test5
RUN_ROOT="${OUTPUT_ROOT}/lora/temporal_causal_count100"
CASE_LIST="${RUN_ROOT}/case_0613pybullet_sample_001460_w002.txt"

mkdir -p "${RUN_ROOT}/heatmaps" "${OUTPUT_ROOT}/logs"
printf '%s\n' \
  /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_001460_w002.json \
  > "${CASE_LIST}"

export CUDA_VISIBLE_DEVICES="${GPU}"
cd "${DIFFTRACK}"
ATTENTION_NOISE_MODE=probability_temporal_causal \
ATTENTION_NOISE_ALPHA=0 \
ATTENTION_MASK_LATENT_FRAMES=7 \
QK_ATTENTION_CAPTURE_ROOT="${RUN_ROOT}/heatmaps" \
QK_ATTENTION_CAPTURE_STEP=39 \
QK_ATTENTION_CAPTURE_MODEL=lora \
"${PYTHON}" "${WORKER}" \
  --model lora \
  --input-json-list "${CASE_LIST}" \
  --output-root "${RUN_ROOT}/videos" \
  --shard-index 0 \
  --num-shards 1 \
  --ranking-pool all720 \
  --extreme-count 100

printf 'gpu=%s\nmodel=lora\nintervention=temporal_causal\ncase=0613pybullet_sample_001460_w002\ncompleted=%s\nsteps=40\nframes=49\n' \
  "${GPU}" "$(date -u +%FT%TZ)" \
  > "${OUTPUT_ROOT}/logs/lora_temporal_causal_case001460.complete"
