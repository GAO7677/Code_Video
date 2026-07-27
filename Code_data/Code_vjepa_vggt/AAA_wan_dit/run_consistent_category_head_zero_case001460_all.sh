#!/usr/bin/env bash
set -euo pipefail

# Run:
# GPU_IDS=3,6 bash run_consistent_category_head_zero_case001460_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/agent-data/outputs/wan_dit_consistent_category_head_ablation/case001460_wan_lora}"
GALLERY_OUTPUT="${GALLERY_OUTPUT:-/data/gaoya/agent-data/outputs/wan_dit_allblock_head_roles/case001460_consistent_category_head_zero_wan_lora}"
CLASSIFICATION_METADATA=/data/gaoya/agent-data/outputs/wan_dit_allblock_head_roles/case001460_latent_aligned_wan_lora/metadata.json
BASELINE_VIDEO=/data/gaoya/agent-data/outputs/wan_dit_ball_query_attention/case001460_frame08/generated/wan_lora/0613pybullet_sample_001460_w002.mp4
SOURCE_JSON=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_001460_w002.json
VERIFY="${SCRIPT_DIR}/verify_consistent_category_head_zero.py"
RUN_ONE="${SCRIPT_DIR}/run_consistent_category_head_zero_case001460_one.sh"
BUILD_GALLERY="${SCRIPT_DIR}/build_consistent_category_head_zero_gallery.py"
GPU_IDS="${GPU_IDS:-3,6}"
CATEGORIES=(S ST T P C G)

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}/logs" "${OUTPUT_BASE}/state"

run_worker() {
  local worker_id="$1"
  local gpu="$2"
  local index
  for index in "${!CATEGORIES[@]}"; do
    (( index % ${#GPUS[@]} == worker_id )) || continue
    local category="${CATEGORIES[index]}"
    local tag="self_attn_consistent_head_zero_category_${category,,}"
    local output_root="${OUTPUT_BASE}/${tag}"
    local log="${OUTPUT_BASE}/logs/${category}.log"
    local complete="${OUTPUT_BASE}/state/${category}.complete"
    if "${PYTHON}" "${VERIFY}" \
      --output-root "${output_root}" \
      --classification-metadata "${CLASSIFICATION_METADATA}" \
      --category "${category}" > "${log}.precheck" 2>&1; then
      cp "${log}.precheck" "${log}"
      printf 'category=%s\ngpu=%s\nreused_existing=true\n' \
        "${category}" "${gpu}" > "${complete}"
      echo "[worker ${worker_id}] reuse ${category}"
      continue
    fi
    echo "[worker ${worker_id}] run ${category} on GPU ${gpu}"
    OUTPUT_BASE="${OUTPUT_BASE}" \
      bash "${RUN_ONE}" "${category}" "${gpu}" > "${log}" 2>&1
    "${PYTHON}" "${VERIFY}" \
      --output-root "${output_root}" \
      --classification-metadata "${CLASSIFICATION_METADATA}" \
      --category "${category}" >> "${log}" 2>&1
    printf 'category=%s\ngpu=%s\n' "${category}" "${gpu}" > "${complete}"
    echo "[worker ${worker_id}] complete ${category}"
  done
}

pids=()
for worker_id in "${!GPUS[@]}"; do
  run_worker "${worker_id}" "${GPUS[worker_id]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo "At least one category failed; inspect ${OUTPUT_BASE}/logs" >&2
  exit "${status}"
fi

"${PYTHON}" "${BUILD_GALLERY}" \
  --root "${OUTPUT_BASE}" \
  --classification-metadata "${CLASSIFICATION_METADATA}" \
  --baseline-video "${BASELINE_VIDEO}" \
  --source-json "${SOURCE_JSON}" \
  --output "${GALLERY_OUTPUT}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_BASE}/state/all.complete"
echo "gallery=${GALLERY_OUTPUT}/index.html"
