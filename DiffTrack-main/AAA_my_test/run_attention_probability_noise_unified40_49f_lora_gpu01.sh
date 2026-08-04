#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 GPU_ID ALPHA ALPHA_TAG" >&2
  exit 2
fi

GPU_ID="$1"
ALPHA="$2"
ALPHA_TAG="$3"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${DIFFTRACK}/AAA_my_test/run_pck_step_adaptive_qk_probability_noise_49f_worker.py"
TEST_LIST="/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/attention_probability_noise_unified_steps40_frames49_test5"
LORA_ROOT="${OUTPUT_ROOT}/lora"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${DIFFTRACK}"

for count in 100 30; do
  run_root="${LORA_ROOT}/alpha${ALPHA_TAG}_count${count}"
  mkdir -p "${run_root}"
  echo "[unified-lora] gpu=${GPU_ID} alpha=${ALPHA} count=${count} steps=40 frames=49"
  ATTENTION_NOISE_MODE=probability_additive \
  ATTENTION_NOISE_ALPHA="${ALPHA}" \
  ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_CAPTURE_ROOT="${run_root}" \
  QK_ATTENTION_CAPTURE_STEP=39 \
  QK_ATTENTION_CAPTURE_MODEL=lora \
  "${PYTHON}" "${WORKER}" \
    --model lora \
    --input-json-list "${TEST_LIST}" \
    --output-root "${run_root}/videos" \
    --shard-index 0 \
    --num-shards 1 \
    --ranking-pool all720 \
    --extreme-count "${count}"
done

printf 'gpu=%s\nalpha=%s\ncompleted=%s\nsteps=40\nframes=49\n' \
  "${GPU_ID}" "${ALPHA}" "$(date -u +%FT%TZ)" \
  > "${OUTPUT_ROOT}/logs/lora_gpu${GPU_ID}.complete"
echo "UNIFIED40_49F_LORA_GPU${GPU_ID}_COMPLETE"
