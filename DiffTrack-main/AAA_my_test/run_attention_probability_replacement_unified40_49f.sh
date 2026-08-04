#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 GPU_ID MODEL {zero|uniform}" >&2
  exit 2
fi

GPU="$1"
MODEL="$2"
INTERVENTION="$3"
DIFFTRACK=/home/gaoya/Code_Video/DiffTrack-main
EXPERIMENT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER="${DIFFTRACK}/AAA_my_test/run_pck_step_adaptive_attention_replacement_49f_worker.py"
TEST_LIST=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/attention_probability_replacement_steps40_frames49_test5
RUN_ROOT="${OUTPUT_ROOT}/${MODEL}/${INTERVENTION}_count100"
LOG_ROOT="${OUTPUT_ROOT}/logs"

if [[ "${GPU}" == "4" || "${GPU}" == "6" || "${GPU}" == "7" ]]; then
  echo "GPU ${GPU} is reserved and will not be used." >&2
  exit 2
fi
if [[ "${MODEL}" != "baseline" && "${MODEL}" != "lora" && "${MODEL}" != "full_sa" ]]; then
  echo "MODEL must be baseline, lora, or full_sa" >&2
  exit 2
fi
if [[ "${INTERVENTION}" != "zero" && "${INTERVENTION}" != "uniform" ]]; then
  echo "INTERVENTION must be zero or uniform" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

if [[ "${MODEL}" == "full_sa" ]]; then
  cd "${EXPERIMENT}"
  NUM_INFERENCE_STEPS=40 \
  RUN_BASELINE=0 \
  ATTENTION_REPLACEMENTS="${INTERVENTION}" \
  ATTENTION_COUNTS=100 \
  ATTENTION_DIRECTIONS="top bottom" \
  ATTENTION_NOISE_SEED=851 \
  TEST_LIST="${TEST_LIST}" \
  bash run_infer_full_sa_no_object_attention_replacement.sh "${GPU}" "${RUN_ROOT}"
else
  cd "${DIFFTRACK}"
  ATTENTION_NOISE_MODE="probability_${INTERVENTION}" \
  ATTENTION_NOISE_ALPHA=0 \
  ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_CAPTURE_ROOT="${RUN_ROOT}/heatmaps" \
  QK_ATTENTION_CAPTURE_STEP=39 \
  QK_ATTENTION_CAPTURE_MODEL="${MODEL}" \
  "${PYTHON}" "${WORKER}" \
    --model "${MODEL}" \
    --input-json-list "${TEST_LIST}" \
    --output-root "${RUN_ROOT}/videos" \
    --shard-index 0 \
    --num-shards 1 \
    --ranking-pool all720 \
    --extreme-count 100
fi

printf 'gpu=%s\nmodel=%s\nintervention=%s\ncompleted=%s\nsteps=40\nframes=49\n' \
  "${GPU}" "${MODEL}" "${INTERVENTION}" "$(date -u +%FT%TZ)" \
  > "${LOG_ROOT}/${MODEL}_${INTERVENTION}.complete"
echo "ATTENTION_REPLACEMENT_COMPLETE model=${MODEL} intervention=${INTERVENTION} gpu=${GPU}"
