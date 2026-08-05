#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU SEED}"
SEED="${2:?usage: $0 GPU SEED}"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${DIFFTRACK}/AAA_my_test/run_attention_lora_seed_sweep_worker.py"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_case001460"
CASE="0613pybullet_sample_001460_w002"
SID="$(printf '%06d' "${SEED}")"
SEED_ROOT="${ROOT}/seeds/seed_${SID}"
CASE_LIST="${ROOT}/case_list.txt"
BASELINE="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds/seed_${SID}/original.mp4"
PROBE_ROOT="${SEED_ROOT}/probe/captures"

mkdir -p "${ROOT}/logs" "${SEED_ROOT}"
printf '%s\n' "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/${CASE}.json" > "${CASE_LIST}"
[[ -s "${BASELINE}" ]] || { echo "Missing baseline: ${BASELINE}" >&2; exit 1; }
[[ $(find "${PROBE_ROOT}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l) -ge 160 ]] || {
  echo "Incomplete shared probe for seed ${SEED}" >&2; exit 1;
}
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${DIFFTRACK}"

link_baseline() {
  local video_root="$1"
  local target="${video_root}/lora/cases/${CASE}/original.mp4"
  mkdir -p "$(dirname "${target}")"
  [[ -e "${target}" ]] || ln "${BASELINE}" "${target}" 2>/dev/null || cp --reflink=auto "${BASELINE}" "${target}"
}

for SPEC in p95_single_d1:0.95 p99_single_d1:0.99; do
  LABEL="${SPEC%%:*}"; QUANTILE="${SPEC##*:}"
  MASK_ROOT="${SEED_ROOT}/trajectory/${LABEL}/masks"
  RENDER_ROOT="${SEED_ROOT}/trajectory/${LABEL}/overlays"
  mkdir -p "${MASK_ROOT}" "${RENDER_ROOT}"
  MASK_COUNT=$(find "${MASK_ROOT}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  if [[ "${MASK_COUNT}" -lt 160 || ! -f "${RENDER_ROOT}/manifest.json" ]]; then
    "${PYTHON}" AAA_my_test/build_object_query_frozen_trajectory_masks.py \
      --probe-root "${PROBE_ROOT}" --output-root "${MASK_ROOT}" \
      --render-root "${RENDER_ROOT}" --video "${BASELINE}" \
      --quantile "${QUANTILE}" --radius 2 --single-component \
      --removal-dilate-radius 1
  fi
  for STAGE in all_steps steps00_09; do
    RUN_ROOT="${SEED_ROOT}/apply/${LABEL}/${STAGE}"
    [[ -f "${RUN_ROOT}/complete" ]] && continue
    VIDEO_ROOT="${RUN_ROOT}/videos"; CAPTURE_ROOT="${RUN_ROOT}/captures"; OVERLAY_ROOT="${RUN_ROOT}/overlays"
    ACTIVE_END=39; SUFFIX=steps_00_40
    [[ "${STAGE}" == "steps00_09" ]] && { ACTIVE_END=9; SUFFIX=steps_00_10; }
    mkdir -p "${CAPTURE_ROOT}" "${OVERLAY_ROOT}"
    link_baseline "${VIDEO_ROOT}"
    ATTENTION_NOISE_MODE=probability_object_query_frozen_trajectory \
    ATTENTION_NOISE_ALPHA=0 ATTENTION_NOISE_SEED="${SEED}" QK_ATTENTION_NOISE_SEED="${SEED}" \
    ATTENTION_GROUP_FILTER=top ATTENTION_CFG_BRANCH_MODE=both \
    ATTENTION_MASK_LATENT_FRAMES=13 ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
    OBJECT_GROUP_ACTIVE_STEP_END="${ACTIVE_END}" OBJECT_GROUP_EXPECTED_HEADS=100 \
    OBJECT_TRAJECTORY_MASK_ROOT="${MASK_ROOT}" \
    OBJECT_TRAJECTORY_APPLY_CAPTURE_ROOT="${CAPTURE_ROOT}" \
      "${PYTHON}" "${WORKER}" --seed "${SEED}" \
        --profile object_query_main_component --stage "${STAGE}" --ranking-criterion pck32 \
        --input-json-list "${CASE_LIST}" --output-root "${VIDEO_ROOT}"
    VIDEO="${VIDEO_ROOT}/lora/cases/${CASE}/top100_${SUFFIX}.mp4"
    [[ -s "${VIDEO}" ]] || { echo "Missing dilate1 video: ${VIDEO}" >&2; exit 1; }
    "${PYTHON}" AAA_my_test/render_object_query_frozen_trajectory_apply.py \
      --capture-root "${CAPTURE_ROOT}" --video "${VIDEO}" --output-root "${OVERLAY_ROOT}"
    printf 'gpu=%s\nseed=%s\nquantile=%s\nscheme=single_component_dilate1\nstage=%s\ncompleted=%s\n' \
      "${GPU}" "${SEED}" "${QUANTILE}" "${STAGE}" "$(date -u +%FT%TZ)" > "${RUN_ROOT}/complete"
  done
done
printf 'gpu=%s\nseed=%s\nscheme=single_component_dilate1_p95_p99\ncompleted=%s\n' \
  "${GPU}" "${SEED}" "$(date -u +%FT%TZ)" > "${SEED_ROOT}/dilate1_p95_p99_complete"
