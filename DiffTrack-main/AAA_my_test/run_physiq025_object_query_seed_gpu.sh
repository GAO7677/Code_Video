#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU SEED}"
SEED="${2:?usage: $0 GPU SEED}"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${DIFFTRACK}/AAA_my_test/run_attention_lora_seed_sweep_worker.py"
RENDER="${DIFFTRACK}/AAA_my_test/render_object_query_continuity_overlay.py"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_object_query_physiq025_10seed"
INPUT_ROOT="/data/gaoya/agent-data/inputs/object_query_physiq025"
REGION_CACHE="/data/gaoya/agent-data/cache/object_query_physiq025_regions/case_physiq025_object_query"
CASE="physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed"
CASE_JSON="/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/${CASE}.json"
CASE_LIST="${ROOT}/case_list.txt"
SEED_ROOT="${ROOT}/seeds/seed_$(printf '%06d' "${SEED}")"
BASELINE="${SEED_ROOT}/original.mp4"

mkdir -p "${ROOT}/logs" "${SEED_ROOT}"
printf '%s\n' "${CASE_JSON}" > "${CASE_LIST}"
cp "${INPUT_ROOT}/seeds.txt" "${ROOT}/seeds.txt"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${DIFFTRACK}"

run_one() {
  local method="$1" stage="$2" capture_step="$3" mode="$4" profile="$5"
  local worker_stage="${stage}"
  [[ "${method}" == "identity" ]] && worker_stage="all_steps"
  local run_root="${SEED_ROOT}/${method}/${stage}"
  local video_root="${run_root}/videos"
  local capture_root="${run_root}/captures"
  local overlay_root="${run_root}/overlays"
  local suffix="steps_00_40"
  [[ "${stage}" == "steps00_09" ]] && suffix="steps_00_10"
  local case_root="${video_root}/lora/cases/${CASE}"
  local original="${case_root}/original.mp4"
  local generated="${case_root}/top100_${suffix}.mp4"
  mkdir -p "${case_root}" "${capture_root}" "${overlay_root}"
  if [[ -s "${BASELINE}" && ! -e "${original}" ]]; then
    ln "${BASELINE}" "${original}" 2>/dev/null || cp --reflink=auto "${BASELINE}" "${original}"
  fi
  ATTENTION_NOISE_MODE="${mode}" \
  ATTENTION_NOISE_ALPHA=0 \
  ATTENTION_NOISE_SEED="${SEED}" \
  QK_ATTENTION_NOISE_SEED="${SEED}" \
  ATTENTION_GROUP_FILTER=top \
  ATTENTION_CFG_BRANCH_MODE=both \
  ATTENTION_MASK_LATENT_FRAMES=13 \
  ATTENTION_MASK_CONTEXT_LATENT_FRAMES=2 \
  OBJECT_CONTINUITY_HIGH_QUANTILE=0.90 \
  OBJECT_CONTINUITY_NEIGHBOR_RADIUS=1 \
  OBJECT_CONTINUITY_MAIN_COMPONENT_TOPK=5 \
  OBJECT_CONTINUITY_CAPTURE_ROOT="${capture_root}" \
  OBJECT_CONTINUITY_CAPTURE_STEP="${capture_step}" \
  OBJECT_QUERY_REGION_CACHE="${REGION_CACHE}" \
  QK_ATTENTION_CAPTURE_CASE="${CASE}" \
    "${PYTHON}" "${WORKER}" \
      --seed "${SEED}" \
      --profile "${profile}" \
      --stage "${worker_stage}" \
      --ranking-criterion pck32 \
      --input-json-list "${CASE_LIST}" \
      --output-root "${video_root}"
  [[ -s "${generated}" ]] || { echo "Missing generated video: ${generated}" >&2; exit 1; }
  if [[ ! -s "${BASELINE}" ]]; then
    [[ -s "${original}" ]] || { echo "Missing original video: ${original}" >&2; exit 1; }
    ln "${original}" "${BASELINE}" 2>/dev/null || cp --reflink=auto "${original}" "${BASELINE}"
  fi
  local render_video="${generated}"
  [[ "${method}" == "identity" ]] && render_video="${BASELINE}"
  "${PYTHON}" "${RENDER}" \
    --capture-root "${capture_root}" \
    --video "${render_video}" \
    --output-root "${overlay_root}"
  printf 'gpu=%s\nseed=%s\nmethod=%s\nstage=%s\ncompleted=%s\n' \
    "${GPU}" "${SEED}" "${method}" "${stage}" "$(date -u +%FT%TZ)" \
    > "${run_root}/complete"
}

run_one identity step39 39 probability_object_query_identity object_query_identity
run_one identity step09 9 probability_object_query_identity object_query_identity
run_one old all_steps 39 probability_object_query_continuity object_query_continuity
run_one old steps00_09 9 probability_object_query_continuity object_query_continuity
run_one new all_steps 39 probability_object_query_main_component_continuity object_query_main_component
run_one new steps00_09 9 probability_object_query_main_component_continuity object_query_main_component
printf 'gpu=%s\nseed=%s\ncompleted=%s\n' \
  "${GPU}" "${SEED}" "$(date -u +%FT%TZ)" > "${SEED_ROOT}/complete"
