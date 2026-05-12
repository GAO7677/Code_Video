#!/usr/bin/env bash
# 用途：分片启动全部物体的 counterfactual 生成任务。
set -euo pipefail

WAN_PY="${WAN_PY:-/data/gaoya/miniconda3/envs/wan/bin/python}"
SCRIPT="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/try1_physxnet_articulation_mpm0417.py"
CAPTION_SCRIPT="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generate_video_captions.py"
PHYSX_ROOT="${PHYSX_ROOT:-/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet}"
VERSION="${VERSION:-version_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases}"
FINALJSON_DIR="${PHYSX_ROOT}/${VERSION}/finaljson"

GPU_ID="${GPU_ID:-0}"
PROCS_PER_GPU="${PROCS_PER_GPU:-10}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
MAX_OBJECTS="${MAX_OBJECTS:-0}"
GENERATE_CAPTIONS="${GENERATE_CAPTIONS:-0}"

LOG_ROOT="${OUTPUT_ROOT}/_counterfactual_logs"
RUN_TAG="gpu${GPU_ID}_shard${SHARD_INDEX}of${SHARD_COUNT}"
WORK_ROOT="${LOG_ROOT}/${RUN_TAG}"
WORKER_LOG_DIR="${WORK_ROOT}/workers"
FAILED_LIST="${WORK_ROOT}/failed_object_ids.txt"
DONE_LIST="${WORK_ROOT}/done_object_ids.txt"
CAPTION_FAILED_LIST="${WORK_ROOT}/caption_failed_object_ids.txt"

CASE_INDICES=(0 1 2 3 5 6 7 100 101 102 900 901)
CASE_INDICES_COUNT_01=(0 1 2 3 5 6 7 900 901)

mkdir -p "${OUTPUT_ROOT}" "${WORK_ROOT}" "${WORKER_LOG_DIR}"
: > "${FAILED_LIST}"
: > "${CAPTION_FAILED_LIST}"
: > "${DONE_LIST}"

if [[ ! -x "${WAN_PY}" ]]; then
  echo "ERROR: wan python not found: ${WAN_PY}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: generator script not found: ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${FINALJSON_DIR}" ]]; then
  echo "ERROR: finaljson dir not found: ${FINALJSON_DIR}" >&2
  exit 1
fi
if ! [[ "${SHARD_INDEX}" =~ ^[0-9]+$ && "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: invalid shard config SHARD_INDEX=${SHARD_INDEX} SHARD_COUNT=${SHARD_COUNT}" >&2
  exit 1
fi
if (( SHARD_INDEX >= SHARD_COUNT )); then
  echo "ERROR: SHARD_INDEX must be < SHARD_COUNT" >&2
  exit 1
fi

mapfile -t ALL_OBJECT_IDS < <(
  find "${FINALJSON_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%f\n' \
    | while read -r filename; do
        echo "${filename%.json}"
      done \
    | sort
)

TOTAL_ALL="${#ALL_OBJECT_IDS[@]}"
if [[ "${MAX_OBJECTS}" =~ ^[0-9]+$ ]] && [[ "${MAX_OBJECTS}" -gt 0 ]] && [[ "${MAX_OBJECTS}" -lt "${TOTAL_ALL}" ]]; then
  ALL_OBJECT_IDS=("${ALL_OBJECT_IDS[@]:0:${MAX_OBJECTS}}")
  TOTAL_ALL="${#ALL_OBJECT_IDS[@]}"
fi

OBJECT_IDS=()
for ((global_idx=0; global_idx<TOTAL_ALL; global_idx++)); do
  if (( global_idx % SHARD_COUNT == SHARD_INDEX )); then
    OBJECT_IDS+=("${ALL_OBJECT_IDS[global_idx]}")
  fi
done
TOTAL="${#OBJECT_IDS[@]}"

echo "WAN_PY=${WAN_PY}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "FINALJSON_DIR=${FINALJSON_DIR}"
echo "GPU_ID=${GPU_ID}"
echo "PROCS_PER_GPU=${PROCS_PER_GPU}"
echo "SHARD_INDEX=${SHARD_INDEX}"
echo "SHARD_COUNT=${SHARD_COUNT}"
echo "TOTAL_ALL=${TOTAL_ALL}"
echo "TOTAL_THIS_SHARD=${TOTAL}"
echo "WORK_ROOT=${WORK_ROOT}"

case_index_to_name() {
  local case_idx="$1"
  case "${case_idx}" in
    0) echo "case000_static_center" ;;
    1) echo "case001_static_left" ;;
    2) echo "case002_static_right" ;;
    3) echo "case003_static_highdrop" ;;
    5) echo "case005_entry_left" ;;
    6) echo "case006_entry_right" ;;
    7) echo "case007_entry_fast_center" ;;
    100) echo "case000_static_center_v2" ;;
    101) echo "case001_static_left_v2" ;;
    102) echo "case002_static_right_v2" ;;
    900) echo "case900_random_parabola" ;;
    901) echo "case901_high_drop" ;;
    *) return 1 ;;
  esac
}

case_base_names_for_bucket() {
  local scene_composition="$1"
  local count_bucket="$2"
  local required_cases=("${CASE_INDICES[@]}")
  if [[ "${scene_composition}" == "single_object_preview" && "${count_bucket}" == "count_01" ]]; then
    required_cases=("${CASE_INDICES_COUNT_01[@]}")
  fi
  local case_idx case_name
  for case_idx in "${required_cases[@]}"; do
    case_name="$(case_index_to_name "${case_idx}")" || return 1
    echo "${case_idx}:${case_name}"
  done
}

counterfactual_missing() {
  local object_id="$1"
  local scene_composition="$2"
  local count_bucket="$3"
  local case_name="$4"
  local same_scene_meta="${OUTPUT_ROOT}/train/rigid/${scene_composition}/${count_bucket}/${object_id}__${case_name}__cf_same_scene_neg/metadata.json"
  local no_collision_meta="${OUTPUT_ROOT}/train/rigid/${scene_composition}/${count_bucket}/${object_id}__${case_name}__cf_no_collision_neg/metadata.json"

  [[ -f "${same_scene_meta}" ]] || return 0
  [[ -f "${no_collision_meta}" ]] || return 0
  return 1
}

collect_missing_case_indices_for_bucket() {
  local object_id="$1"
  local scene_composition="$2"
  local count_bucket="$3"
  local item case_idx case_name
  while IFS= read -r item; do
    [[ -z "${item}" ]] && continue
    case_idx="${item%%:*}"
    case_name="${item#*:}"
    if counterfactual_missing "${object_id}" "${scene_composition}" "${count_bucket}" "${case_name}"; then
      echo "${case_idx}"
    fi
  done < <(case_base_names_for_bucket "${scene_composition}" "${count_bucket}")
}

generate_captions_for_object() {
  local object_id="$1"
  local log_path="$2"
  mapfile -t metadata_paths < <(
    find "${OUTPUT_ROOT}/train/rigid" -type f -name metadata.json \
      \( \
        -path "*/${object_id}__case*/*" \
        -o -path "*/invalid_case900_901/${object_id}__case*/*" \
        -o -path "*/invalid_by_qa/*/${object_id}__case*/*" \
        -o -path "*/_qa_invalid/*/${object_id}__case*/*" \
      \) \
      | sort
  )
  if [[ "${#metadata_paths[@]}" -eq 0 ]]; then
    echo "CAPTION skip object_id=${object_id} reason=no_metadata_found" | tee -a "${log_path}"
    return 0
  fi
  "${WAN_PY}" "${CAPTION_SCRIPT}" --roots "${metadata_paths[@]}" --include_invalid 2>&1 | tee -a "${log_path}"
}

run_bucket_if_missing() {
  local object_id="$1"
  local target_count="$2"
  local scene_composition="$3"
  local count_bucket="$4"
  local log_path="$5"
  local missing_case_indices=()

  mapfile -t missing_case_indices < <(
    collect_missing_case_indices_for_bucket "${object_id}" "${scene_composition}" "${count_bucket}"
  )
  if [[ "${#missing_case_indices[@]}" -eq 0 ]]; then
    echo "SKIP complete object_id=${object_id} bucket=${scene_composition}/${count_bucket}" | tee -a "${log_path}"
    return 0
  fi

  echo "RUN object_id=${object_id} gpu=${GPU_ID} bucket=${scene_composition}/${count_bucket} missing_cases=${missing_case_indices[*]}" | tee -a "${log_path}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  EGL_DEVICE_ID="${GPU_ID}" \
  PYOPENGL_PLATFORM=egl \
  PYTHONUNBUFFERED=1 \
  "${WAN_PY}" "${SCRIPT}" \
    --physx_root "${PHYSX_ROOT}" \
    --version "${VERSION}" \
    --object_id "${object_id}" \
    --output_root "${OUTPUT_ROOT}" \
    --run_genesis \
    --generate_all_count_motion_cases \
    --case_index_filter "${missing_case_indices[@]}" \
    --rigid_count_filter "${target_count}" \
    --prefer_existing_runtime_meshes \
    --enable_counterfactual_cases \
    --counterfactual_only \
    --dt 0.003 \
    --substeps 40 \
    --ball_posx 0.03 \
    --steps 12 \
    --fps 12 \
    --simulator_mode rigid \
    2>&1 | tee -a "${log_path}"
}

process_object() {
  local object_id="$1"
  local idx="$2"
  local log_path="${WORK_ROOT}/${object_id}.log"
  : > "${log_path}"

  echo "==> [${idx}/${TOTAL}] object_id=${object_id} gpu=${GPU_ID}" | tee -a "${log_path}"

  set +e
  run_bucket_if_missing "${object_id}" 1 "single_object_preview" "count_01" "${log_path}" &&
  run_bucket_if_missing "${object_id}" 2 "interaction_pair_plus_dynamic" "count_02" "${log_path}" &&
  run_bucket_if_missing "${object_id}" 3 "interaction_pair_plus_dynamic" "count_03_04" "${log_path}" &&
  run_bucket_if_missing "${object_id}" 4 "interaction_pair_plus_dynamic" "count_03_04" "${log_path}"
  local status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "FAILED object_id=${object_id} gpu=${GPU_ID}" | tee -a "${log_path}"
    echo "${object_id}" >> "${FAILED_LIST}"
    return 0
  fi

  if [[ "${GENERATE_CAPTIONS}" == "1" ]]; then
    set +e
    generate_captions_for_object "${object_id}" "${log_path}"
    local caption_status=$?
    set -e
    if [[ "${caption_status}" -ne 0 ]]; then
      echo "CAPTION FAILED object_id=${object_id}" | tee -a "${log_path}"
      echo "${object_id}" >> "${CAPTION_FAILED_LIST}"
    fi
  fi

  echo "${object_id}" >> "${DONE_LIST}"
}

worker_main() {
  local slot_idx="$1"
  local worker_log="${WORKER_LOG_DIR}/worker_${slot_idx}.log"
  : > "${worker_log}"
  echo "WORKER slot=${slot_idx} gpu=${GPU_ID} start" | tee -a "${worker_log}"
  local idx
  for ((idx=slot_idx; idx<TOTAL; idx+=PROCS_PER_GPU)); do
    local object_id="${OBJECT_IDS[idx]}"
    echo "WORKER slot=${slot_idx} pick index=$((idx+1)) object_id=${object_id}" | tee -a "${worker_log}"
    process_object "${object_id}" "$((idx+1))" 2>&1 | tee -a "${worker_log}"
  done
  echo "WORKER slot=${slot_idx} gpu=${GPU_ID} done" | tee -a "${worker_log}"
}

if [[ "${TOTAL}" -eq 0 ]]; then
  echo "No objects assigned to this shard."
  exit 0
fi

PIDS=()
for ((slot_idx=0; slot_idx<PROCS_PER_GPU; slot_idx++)); do
  worker_main "${slot_idx}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "Shard finished. GPU=${GPU_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} done=$(wc -l < "${DONE_LIST}") failed=$(wc -l < "${FAILED_LIST}")"
exit "${status}"
