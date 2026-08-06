#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU SEED}"
SEED="${2:?usage: $0 GPU SEED}"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${DIFFTRACK}/AAA_my_test/run_attention_lora_seed_sweep_worker.py"
BUILD_MASKS="${DIFFTRACK}/AAA_my_test/build_object_query_frozen_trajectory_masks.py"
RENDER_APPLY="${DIFFTRACK}/AAA_my_test/render_object_query_frozen_trajectory_apply.py"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_physiq025_2seed"
SOURCE_ROOT="/data/gaoya/agent-data/outputs/attention_lora_object_query_physiq025_10seed"
REGION_CACHE="/data/gaoya/agent-data/cache/object_query_physiq025_regions/case_physiq025_object_query"
CASE="physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed"
CASE_JSON="/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/${CASE}.json"
SID="$(printf '%06d' "${SEED}")"
SEED_ROOT="${ROOT}/seeds/seed_${SID}"
CASE_LIST="${ROOT}/case_list.txt"
BASELINE="${SOURCE_ROOT}/seeds/seed_${SID}/original.mp4"
TOP100_PROBE_ROOT="${SEED_ROOT}/probe/captures"
TOP30_PROBE_ROOT="${SEED_ROOT}/probe_top30/captures"

mkdir -p "${ROOT}/logs" "${SEED_ROOT}"
printf '%s\n' "${CASE_JSON}" > "${CASE_LIST}"
printf '13161\n16342\n' > "${ROOT}/seeds.txt"
[[ -s "${BASELINE}" ]] || { echo "Missing PhysIQ baseline: ${BASELINE}" >&2; exit 1; }
[[ -s "${REGION_CACHE}/regions.npz" ]] || { echo "Missing SAM2 region cache: ${REGION_CACHE}" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${DIFFTRACK}"

link_baseline() {
  local video_root="$1"
  local target="${video_root}/lora/cases/${CASE}/original.mp4"
  mkdir -p "$(dirname "${target}")"
  [[ -e "${target}" ]] || ln "${BASELINE}" "${target}" 2>/dev/null || cp --reflink=auto "${BASELINE}" "${target}"
}

run_probe() {
  local count="$1" probe_name="$2" probe_root="$3"
  local probe_video_root="${SEED_ROOT}/${probe_name}/videos"
  local complete="${SEED_ROOT}/${probe_name}/complete"
  local capture_count
  mkdir -p "${probe_root}"
  capture_count=$(find "${probe_root}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  if [[ "${capture_count}" -ge 160 && -f "${complete}" ]]; then
    return
  fi
  link_baseline "${probe_video_root}"
  ATTENTION_NOISE_MODE=probability_object_query_trajectory_probe \
  ATTENTION_NOISE_ALPHA=0 ATTENTION_NOISE_SEED="${SEED}" QK_ATTENTION_NOISE_SEED="${SEED}" \
  ATTENTION_EXTREME_COUNT="${count}" ATTENTION_GROUP_FILTER=top ATTENTION_CFG_BRANCH_MODE=both \
  ATTENTION_MASK_LATENT_FRAMES=13 ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
  OBJECT_GROUP_ACTIVE_STEP_END=39 OBJECT_GROUP_EXPECTED_HEADS="${count}" \
  OBJECT_CONTINUITY_HIGH_QUANTILE=0.90 OBJECT_CONTINUITY_NEIGHBOR_RADIUS=2 \
  OBJECT_CONTINUITY_MAIN_COMPONENT_TOPK=5 OBJECT_TRAJECTORY_PROBE_ROOT="${probe_root}" \
  OBJECT_QUERY_REGION_CACHE="${REGION_CACHE}" QK_ATTENTION_CAPTURE_CASE="${CASE}" \
    "${PYTHON}" "${WORKER}" --seed "${SEED}" \
      --profile object_query_main_component --stage all_steps --ranking-criterion pck32 \
      --input-json-list "${CASE_LIST}" --output-root "${probe_video_root}"
  capture_count=$(find "${probe_root}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  [[ "${capture_count}" -ge 160 ]] || { echo "Incomplete ${probe_name}: ${capture_count}/160" >&2; exit 1; }
  printf 'gpu=%s\nseed=%s\nmean_heads=%s\ncompleted=%s\n' \
    "${GPU}" "${SEED}" "${count}" "$(date -u +%FT%TZ)" > "${complete}"
}

run_probe 100 probe "${TOP100_PROBE_ROOT}"
run_probe 30 probe_top30 "${TOP30_PROBE_ROOT}"

for SPEC in \
  'p95:0.95:top100:multi:0:0' \
  'p99:0.99:top100:multi:0:0' \
  'p95_single:0.95:top100:single:0:0' \
  'p99_single:0.99:top100:single:0:0' \
  'p95_single_d1:0.95:top100:single:1:0' \
  'p99_single_d1:0.99:top100:single:1:0' \
  'p95_single_bt3_d1:0.95:top30:single:1:3' \
  'p99_single_bt3_d1:0.99:top30:single:1:3'; do
  IFS=: read -r LABEL QUANTILE PROBE_KIND COMPONENT DILATE BACKTRACK <<< "${SPEC}"
  PROBE_ROOT="${TOP100_PROBE_ROOT}"
  [[ "${PROBE_KIND}" == top30 ]] && PROBE_ROOT="${TOP30_PROBE_ROOT}"
  MASK_ROOT="${SEED_ROOT}/trajectory/${LABEL}/masks"
  TRAJECTORY_RENDER="${SEED_ROOT}/trajectory/${LABEL}/overlays"
  mkdir -p "${MASK_ROOT}" "${TRAJECTORY_RENDER}"
  MASK_ARGS=(
    --probe-root "${PROBE_ROOT}" --output-root "${MASK_ROOT}"
    --render-root "${TRAJECTORY_RENDER}" --video "${BASELINE}"
    --quantile "${QUANTILE}" --radius 2
  )
  [[ "${COMPONENT}" == single ]] && MASK_ARGS+=(--single-component)
  [[ "${DILATE}" -gt 0 ]] && MASK_ARGS+=(--removal-dilate-radius "${DILATE}")
  [[ "${BACKTRACK}" -gt 0 ]] && MASK_ARGS+=(--backtrack-frames "${BACKTRACK}")
  MASK_COUNT=$(find "${MASK_ROOT}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  if [[ "${MASK_COUNT}" -lt 160 || ! -f "${TRAJECTORY_RENDER}/manifest.json" ]]; then
    "${PYTHON}" "${BUILD_MASKS}" "${MASK_ARGS[@]}"
  fi

  for STAGE in all_steps steps00_09; do
    RUN_ROOT="${SEED_ROOT}/apply/${LABEL}/${STAGE}"
    [[ -f "${RUN_ROOT}/complete" ]] && continue
    VIDEO_ROOT="${RUN_ROOT}/videos"
    CAPTURE_ROOT="${RUN_ROOT}/captures"
    OVERLAY_ROOT="${RUN_ROOT}/overlays"
    ACTIVE_END=39
    SUFFIX=steps_00_40
    [[ "${STAGE}" == steps00_09 ]] && { ACTIVE_END=9; SUFFIX=steps_00_10; }
    mkdir -p "${CAPTURE_ROOT}" "${OVERLAY_ROOT}"
    link_baseline "${VIDEO_ROOT}"
    ATTENTION_NOISE_MODE=probability_object_query_frozen_trajectory \
    ATTENTION_NOISE_ALPHA=0 ATTENTION_NOISE_SEED="${SEED}" QK_ATTENTION_NOISE_SEED="${SEED}" \
    ATTENTION_EXTREME_COUNT=100 ATTENTION_GROUP_FILTER=top ATTENTION_CFG_BRANCH_MODE=both \
    ATTENTION_MASK_LATENT_FRAMES=13 ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
    OBJECT_GROUP_ACTIVE_STEP_END="${ACTIVE_END}" OBJECT_GROUP_EXPECTED_HEADS=100 \
    OBJECT_TRAJECTORY_MASK_ROOT="${MASK_ROOT}" \
    OBJECT_TRAJECTORY_APPLY_CAPTURE_ROOT="${CAPTURE_ROOT}" \
    OBJECT_QUERY_REGION_CACHE="${REGION_CACHE}" QK_ATTENTION_CAPTURE_CASE="${CASE}" \
      "${PYTHON}" "${WORKER}" --seed "${SEED}" \
        --profile object_query_main_component --stage "${STAGE}" --ranking-criterion pck32 \
        --input-json-list "${CASE_LIST}" --output-root "${VIDEO_ROOT}"
    VIDEO="${VIDEO_ROOT}/lora/cases/${CASE}/top100_${SUFFIX}.mp4"
    [[ -s "${VIDEO}" ]] || { echo "Missing intervention video: ${VIDEO}" >&2; exit 1; }
    "${PYTHON}" "${RENDER_APPLY}" \
      --capture-root "${CAPTURE_ROOT}" --video "${VIDEO}" --output-root "${OVERLAY_ROOT}"
    printf 'gpu=%s\nseed=%s\nlabel=%s\nquantile=%s\nprobe=%s\nstage=%s\ncompleted=%s\n' \
      "${GPU}" "${SEED}" "${LABEL}" "${QUANTILE}" "${PROBE_KIND}" "${STAGE}" \
      "$(date -u +%FT%TZ)" > "${RUN_ROOT}/complete"
  done
done

printf 'gpu=%s\nseed=%s\ncase=%s\nvariants=8\nstages=2\ncompleted=%s\n' \
  "${GPU}" "${SEED}" "${CASE}" "$(date -u +%FT%TZ)" > "${SEED_ROOT}/complete"
