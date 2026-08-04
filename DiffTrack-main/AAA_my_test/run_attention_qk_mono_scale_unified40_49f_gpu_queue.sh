#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 GPU_ID" >&2
  exit 2
fi

GPU="$1"
PROFILE="${MONO_SCALE_PROFILE:-${GPU}}"
CAPTURE_PER_HEAD="${ATTENTION_MONO_CAPTURE_PER_HEAD:-1}"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
EXPERIMENT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${DIFFTRACK}/AAA_my_test/run_pck_step_adaptive_qk_probability_noise_49f_worker.py"
TEST_LIST="${ATTENTION_MONO_TEST_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
OUTPUT_ROOT="${ATTENTION_MONO_OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/attention_probability_mono_scale_steps40_frames49_test5}"
BASELINE_ROOT="${OUTPUT_ROOT}/baseline"
LORA_ROOT="${OUTPUT_ROOT}/lora"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${BASELINE_ROOT}" "${LORA_ROOT}" "${LOG_ROOT}"
if [[ ! -s "${TEST_LIST}" ]]; then
  echo "Input case list is missing or empty: ${TEST_LIST}" >&2
  exit 2
fi
CAPTURE_CASE="${ATTENTION_MONO_CAPTURE_CASE:-}"
if [[ -z "${CAPTURE_CASE}" ]]; then
  mapfile -t INPUT_CASES < <(sed '/^[[:space:]]*$/d' "${TEST_LIST}")
  if [[ ${#INPUT_CASES[@]} -eq 1 ]]; then
    CAPTURE_CASE="$(basename "${INPUT_CASES[0]}" .json)"
  else
    CAPTURE_CASE="case"
  fi
fi
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
  echo "[mono-scale] gpu=${GPU} profile=${PROFILE} model=${model} alpha=${alpha} count=${count} steps=40 frames=49 per_head=${CAPTURE_PER_HEAD}"
  ATTENTION_NOISE_MODE=probability_mono_scale \
  ATTENTION_NOISE_ALPHA="${alpha}" \
  ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_NOISE_SEED=851 \
  QK_ATTENTION_CAPTURE_ROOT="${run_root}" \
  QK_ATTENTION_CAPTURE_STEP=39 \
  QK_ATTENTION_CAPTURE_MODEL="${model}" \
  QK_ATTENTION_CAPTURE_CASE="${CAPTURE_CASE}" \
  QK_ATTENTION_CAPTURE_PER_HEAD="${CAPTURE_PER_HEAD}" \
  QK_ATTENTION_CAPTURE_SMALL_SIZE=416 \
  QK_ATTENTION_CAPTURE_LATENT_FRAMES=13 \
  "${PYTHON}" "${WORKER}" \
    --model "${model}" \
    --input-json-list "${TEST_LIST}" \
    --output-root "${run_root}/videos" \
    --shard-index 0 \
    --num-shards 1 \
    --ranking-pool all720 \
    --extreme-count "${count}"
}

case "${PROFILE}" in
  alpha090)
    run_wan baseline 0.9 090 30 "${BASELINE_ROOT}"
    run_wan lora 0.9 090 30 "${LORA_ROOT}"
    run_wan baseline 0.9 090 100 "${BASELINE_ROOT}"
    run_wan lora 0.9 090 100 "${LORA_ROOT}"
    ;;
  lora_alpha030_count100)
    run_wan lora 0.3 030 100 "${LORA_ROOT}"
    ;;
  lora_alpha060_count100)
    run_wan lora 0.6 060 100 "${LORA_ROOT}"
    ;;
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
  *)
    echo "Profile ${PROFILE} is not assigned; expected one of 0,1,2,3,alpha090,lora_alpha030_count100,lora_alpha060_count100" >&2
    exit 2
    ;;
esac

printf 'gpu=%s\nprofile=%s\ncompleted=%s\nsteps=40\nframes=49\nper_head=%s\n' \
  "${GPU}" "${PROFILE}" "$(date -u +%FT%TZ)" "${CAPTURE_PER_HEAD}" > "${LOG_ROOT}/gpu${GPU}.complete"
echo "MONO_SCALE_QK_49F_GPU${GPU}_COMPLETE"
