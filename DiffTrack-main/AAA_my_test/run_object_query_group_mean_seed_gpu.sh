#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU SEED STAGE}"
SEED="${2:?usage: $0 GPU SEED STAGE}"
STAGE="${3:?usage: $0 GPU SEED STAGE}"
[[ "${STAGE}" == "all_steps" || "${STAGE}" == "steps00_09" ]] || exit 2

DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${DIFFTRACK}/AAA_my_test/run_attention_lora_seed_sweep_worker.py"
RENDER="${DIFFTRACK}/AAA_my_test/render_object_query_group_mean_overlay.py"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_object_query_group_mean_case001460"
CASE="0613pybullet_sample_001460_w002"
CASE_LIST="${ROOT}/case_list.txt"
RUN_ROOT="${ROOT}/seeds/seed_$(printf '%06d' "${SEED}")/${STAGE}"
VIDEO_ROOT="${RUN_ROOT}/videos"
CAPTURE_ROOT="${RUN_ROOT}/captures"
OVERLAY_ROOT="${RUN_ROOT}/overlays"
STEP=39
ACTIVE_END=39
SUFFIX=steps_00_40
[[ "${STAGE}" == "steps00_09" ]] && { STEP=9; ACTIVE_END=9; SUFFIX=steps_00_10; }

mkdir -p "${ROOT}/logs" "${RUN_ROOT}" "${CAPTURE_ROOT}" "${OVERLAY_ROOT}"
printf '%s\n' "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/${CASE}.json" > "${CASE_LIST}"
BASELINE="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds/seed_$(printf '%06d' "${SEED}")/original.mp4"
[[ -s "${BASELINE}" ]] || { echo "Missing Wan+LoRA seed baseline: ${BASELINE}" >&2; exit 1; }
TARGET="${VIDEO_ROOT}/lora/cases/${CASE}/original.mp4"
mkdir -p "$(dirname "${TARGET}")"
[[ -e "${TARGET}" ]] || ln "${BASELINE}" "${TARGET}" 2>/dev/null || cp --reflink=auto "${BASELINE}" "${TARGET}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${DIFFTRACK}"
ATTENTION_NOISE_MODE=probability_object_query_group_mean_continuity \
ATTENTION_NOISE_ALPHA=0 \
ATTENTION_NOISE_SEED="${SEED}" \
QK_ATTENTION_NOISE_SEED="${SEED}" \
ATTENTION_GROUP_FILTER=top \
ATTENTION_CFG_BRANCH_MODE=both \
ATTENTION_MASK_LATENT_FRAMES=13 \
ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
OBJECT_GROUP_ACTIVE_STEP_END="${ACTIVE_END}" \
OBJECT_GROUP_EXPECTED_HEADS=100 \
OBJECT_CONTINUITY_HIGH_QUANTILE=0.90 \
OBJECT_CONTINUITY_NEIGHBOR_RADIUS=1 \
OBJECT_CONTINUITY_MAIN_COMPONENT_TOPK=5 \
OBJECT_CONTINUITY_CAPTURE_ROOT="${CAPTURE_ROOT}" \
OBJECT_CONTINUITY_CAPTURE_STEP="${STEP}" \
QK_ATTENTION_CAPTURE_CASE="${CASE}" \
  "${PYTHON}" "${WORKER}" \
    --seed "${SEED}" \
    --profile object_query_main_component \
    --stage "${STAGE}" \
    --ranking-criterion pck32 \
    --input-json-list "${CASE_LIST}" \
    --output-root "${VIDEO_ROOT}"

VIDEO="${VIDEO_ROOT}/lora/cases/${CASE}/top100_${SUFFIX}.mp4"
[[ -s "${VIDEO}" ]] || { echo "Missing generated video: ${VIDEO}" >&2; exit 1; }
"${PYTHON}" "${RENDER}" \
  --capture-root "${CAPTURE_ROOT}" --video "${VIDEO}" --output-root "${OVERLAY_ROOT}"
printf 'gpu=%s\nseed=%s\nstage=%s\ncompleted=%s\n' \
  "${GPU}" "${SEED}" "${STAGE}" "$(date -u +%FT%TZ)" > "${RUN_ROOT}/complete"
