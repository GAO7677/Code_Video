#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 GPU_ID" >&2
  exit 2
fi

GPU="$1"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
EXPERIMENT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${DIFFTRACK}/AAA_my_test/run_pck_step_adaptive_qk_probability_noise_49f_worker.py"
TEST_LIST="/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/attention_probability_noise_unified_steps40_frames49_test5"
BASELINE_ROOT="${OUTPUT_ROOT}/baseline"
LORA_ROOT="${OUTPUT_ROOT}/lora"
FULLSA_ROOT="${OUTPUT_ROOT}/full_sa"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${BASELINE_ROOT}" "${LORA_ROOT}" "${FULLSA_ROOT}" "${LOG_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

run_wan() {
  local model="$1"
  local alpha="$2"
  local tag="$3"
  local count="$4"
  local model_root="$5"
  local run_root="${model_root}/alpha${tag}_count${count}"
  mkdir -p "${run_root}"
  cd "${DIFFTRACK}"
  echo "[unified] gpu=${GPU} model=${model} alpha=${alpha} count=${count} steps=40 frames=49"
  ATTENTION_NOISE_MODE=probability_additive \
  ATTENTION_NOISE_ALPHA="${alpha}" \
  ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_CAPTURE_ROOT="${run_root}" \
  QK_ATTENTION_CAPTURE_STEP=39 \
  QK_ATTENTION_CAPTURE_MODEL="${model}" \
  "${PYTHON}" "${WORKER}" \
    --model "${model}" \
    --input-json-list "${TEST_LIST}" \
    --output-root "${run_root}/videos" \
    --shard-index 0 \
    --num-shards 1 \
    --ranking-pool all720 \
    --extreme-count "${count}"
}

run_fullsa() {
  local run_baseline="$1"
  local alphas="$2"
  local counts="$3"
  local directions="$4"
  cd "${EXPERIMENT}"
  echo "[unified] gpu=${GPU} model=full_sa alpha=${alphas} counts=${counts} directions=${directions} steps=40 frames=49"
  NUM_INFERENCE_STEPS=40 \
  RUN_BASELINE="${run_baseline}" \
  ATTENTION_ALPHAS="${alphas}" \
  ATTENTION_COUNTS="${counts}" \
  ATTENTION_DIRECTIONS="${directions}" \
  ATTENTION_NOISE_SEED=851 \
  TEST_LIST="${TEST_LIST}" \
  bash run_infer_full_sa_no_object_attention_noise.sh "${GPU}" "${FULLSA_ROOT}"
}

case "${GPU}" in
  0)
    run_wan baseline 0.9 090 30 "${BASELINE_ROOT}"
    run_wan lora 0.9 090 30 "${LORA_ROOT}"
    ;;
  1)
    run_wan baseline 0.9 090 100 "${BASELINE_ROOT}"
    run_wan lora 0.9 090 100 "${LORA_ROOT}"
    ;;
  2)
    run_wan baseline 1.5 150 30 "${BASELINE_ROOT}"
    run_wan lora 1.5 150 30 "${LORA_ROOT}"
    ;;
  3)
    run_wan baseline 1.5 150 100 "${BASELINE_ROOT}"
    run_wan lora 1.5 150 100 "${LORA_ROOT}"
    ;;
  4)
    run_fullsa 1 "0.9" "30 100" "top bottom"
    ;;
  5)
    run_fullsa 0 "1.5" "30 100" "top bottom"
    ;;
  *)
    echo "GPU ${GPU} is not assigned; expected one of 0,1,2,3,4,5" >&2
    exit 2
    ;;
esac

printf 'gpu=%s\ncompleted=%s\nsteps=40\nframes=49\n' \
  "${GPU}" "$(date -u +%FT%TZ)" > "${LOG_ROOT}/gpu${GPU}.complete"
echo "UNIFIED40_49F_GPU${GPU}_COMPLETE"
