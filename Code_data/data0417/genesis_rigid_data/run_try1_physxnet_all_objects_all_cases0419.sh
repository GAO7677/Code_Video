#!/usr/bin/env bash
# 该脚本用于遍历 PhysXNet 全部 object_id 生成 rigid-only 全 case 数据；输入为 /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/finaljson 和生成/打 caption 脚本，输出为 /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases 下的样本、caption 与日志。
set -euo pipefail

# 遍历 PhysXNet version_1/finaljson 下的每一个 object_id，
# 对每个 object 生成 rigid-only 的所有 count/motion case。
# 输出目录按训练规范组织：
#   train/rigid/<scene_composition>/<count_bucket>/<object_id>__<case_name>/

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan

SCRIPT="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/try1_physxnet_articulation_mpm0417.py"
CAPTION_SCRIPT="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generate_video_captions.py"
PHYSX_ROOT="/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet"
VERSION="version_1"
OUTPUT_ROOT="/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases"
FINALJSON_DIR="${PHYSX_ROOT}/${VERSION}/finaljson"
LOG_DIR="${OUTPUT_ROOT}/_logs"
FAILED_LIST="${LOG_DIR}/failed_object_ids.txt"
CAPTION_FAILED_LIST="${LOG_DIR}/caption_failed_object_ids.txt"
DONE_LIST="${LOG_DIR}/done_object_ids.txt"
CASE_INDICES=(0 1 2 3 5 6 7 100 101 102 900 901)
CASE_INDICES_COUNT_01=(0 1 2 3 5 6 7 900 901)
GPU_IDS_CSV="${GPU_IDS:-0}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
MAX_OBJECTS="${MAX_OBJECTS:-0}"
WORKER_LOG_DIR="${LOG_DIR}/workers"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${WORKER_LOG_DIR}"
: > "${FAILED_LIST}"
: > "${CAPTION_FAILED_LIST}"
rm -f "${DONE_LIST}"

if [[ ! -d "${FINALJSON_DIR}" ]]; then
  echo "ERROR: finaljson dir not found: ${FINALJSON_DIR}" >&2
  exit 1
fi

mapfile -t OBJECT_IDS < <(
  find "${FINALJSON_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%f\n' \
    | while read -r filename; do
        echo "${filename%.json}"
      done \
    | sort
)
TOTAL="${#OBJECT_IDS[@]}"
if [[ "${MAX_OBJECTS}" =~ ^[0-9]+$ ]] && [[ "${MAX_OBJECTS}" -gt 0 ]] && [[ "${MAX_OBJECTS}" -lt "${TOTAL}" ]]; then
  OBJECT_IDS=("${OBJECT_IDS[@]:0:${MAX_OBJECTS}}")
  TOTAL="${#OBJECT_IDS[@]}"
fi

IFS=',' read -r -a GPU_IDS_ARR <<< "${GPU_IDS_CSV}"
SLOT_GPU_IDS=()
for gpu_id in "${GPU_IDS_ARR[@]}"; do
  gpu_trimmed="$(echo "${gpu_id}" | xargs)"
  [[ -z "${gpu_trimmed}" ]] && continue
  for ((rep=0; rep<PROCS_PER_GPU; rep++)); do
    SLOT_GPU_IDS+=("${gpu_trimmed}")
  done
done
if [[ "${#SLOT_GPU_IDS[@]}" -eq 0 ]]; then
  SLOT_GPU_IDS=("0")
fi
TOTAL_SLOTS="${#SLOT_GPU_IDS[@]}"

echo "PhysXNet finaljson: ${FINALJSON_DIR}"
echo "Total objects: ${TOTAL}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Logs: ${LOG_DIR}"
echo "Default counts: count=1,2,3,4"
echo "Default cases: ${CASE_INDICES[*]}"
echo "GPU_IDS: ${GPU_IDS_CSV}"
echo "PROCS_PER_GPU: ${PROCS_PER_GPU}"
echo "Total worker slots: ${TOTAL_SLOTS}"

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

collect_missing_case_indices_for_bucket() {
  local object_id="$1"
  local scene_composition="$2"
  local count_bucket="$3"
  local case_idx case_name path
  local required_cases=("${CASE_INDICES[@]}")
  if [[ "${scene_composition}" == "single_object_preview" && "${count_bucket}" == "count_01" ]]; then
    # v2 cases are striker-speed variants. count_01 disables the striker, so
    # these cases are intentionally not generated for single-object samples.
    required_cases=("${CASE_INDICES_COUNT_01[@]}")
  fi

  for case_idx in "${required_cases[@]}"; do
    case_name="$(case_index_to_name "${case_idx}")" || return 1
    path="${OUTPUT_ROOT}/train/rigid/${scene_composition}/${count_bucket}/${object_id}__${case_name}/metadata.json"
    if [[ ! -f "${path}" ]]; then
      echo "${case_idx}"
    fi
  done
}

run_bucket_if_missing() {
  local object_id="$1"
  local target_count="$2"
  local scene_composition="$3"
  local count_bucket="$4"
  local log_path="$5"
  local gpu_id="$6"
  local missing_case_indices=()

  mapfile -t missing_case_indices < <(
    collect_missing_case_indices_for_bucket "${object_id}" "${scene_composition}" "${count_bucket}"
  )
  if [[ "${#missing_case_indices[@]}" -eq 0 ]]; then
    echo "SKIP complete object_id=${object_id} target_count=${target_count} bucket=${scene_composition}/${count_bucket}"
    return 0
  fi

  echo "RUN object_id=${object_id} gpu=${gpu_id} target_count=${target_count} bucket=${scene_composition}/${count_bucket} missing_cases=${missing_case_indices[*]}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python3 "${SCRIPT}" \
    --physx_root "${PHYSX_ROOT}" \
    --version "${VERSION}" \
    --object_id "${object_id}" \
    --output_root "${OUTPUT_ROOT}" \
    --run_genesis \
    --generate_all_count_motion_cases \
    --case_index_filter "${missing_case_indices[@]}" \
    --rigid_count_filter "${target_count}" \
    --prefer_existing_runtime_meshes \
    --dt 0.003 \
    --substeps 40 \
    --ball_posx 0.03 \
    --steps 12 \
    --fps 12 \
    --simulator_mode rigid \
    2>&1 | tee -a "${log_path}"
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

  echo "CAPTION object_id=${object_id} files=${#metadata_paths[@]}" | tee -a "${log_path}"
  python3 "${CAPTION_SCRIPT}" \
    --roots "${metadata_paths[@]}" \
    --include_invalid \
    2>&1 | tee -a "${log_path}"
}

process_object() {
  local object_id="$1"
  local idx="$2"
  local gpu_id="$3"
  log_path="${LOG_DIR}/${object_id}.log"
  echo "==> [${idx}/${TOTAL}] object_id=${object_id} gpu=${gpu_id}"

  set +e
  : > "${log_path}"
  run_bucket_if_missing "${object_id}" 1 "single_object_preview" "count_01" "${log_path}" "${gpu_id}" &&
  run_bucket_if_missing "${object_id}" 2 "interaction_pair_plus_dynamic" "count_02" "${log_path}" "${gpu_id}" &&
  run_bucket_if_missing "${object_id}" 3 "interaction_pair_plus_dynamic" "count_03_04" "${log_path}" "${gpu_id}" &&
  run_bucket_if_missing "${object_id}" 4 "interaction_pair_plus_dynamic" "count_03_04" "${log_path}" "${gpu_id}"
  status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "FAILED object_id=${object_id} gpu=${gpu_id}, see ${log_path}" >&2
    echo "${object_id}" >> "${FAILED_LIST}"
    return 0
  fi

  set +e
  generate_captions_for_object "${object_id}" "${log_path}"
  caption_status=$?
  set -e
  if [[ "${caption_status}" -ne 0 ]]; then
    echo "CAPTION FAILED object_id=${object_id} gpu=${gpu_id}, see ${log_path}" >&2
    echo "${object_id}" >> "${CAPTION_FAILED_LIST}"
  fi

  echo "${object_id}" >> "${DONE_LIST}"
}

worker_main() {
  local slot_idx="$1"
  local gpu_id="$2"
  local worker_log="${WORKER_LOG_DIR}/worker_${slot_idx}_gpu${gpu_id}.log"
  : > "${worker_log}"
  echo "WORKER slot=${slot_idx} gpu=${gpu_id} start" | tee -a "${worker_log}"
  local idx
  for ((idx=slot_idx; idx<TOTAL; idx+=TOTAL_SLOTS)); do
    local object_id="${OBJECT_IDS[idx]}"
    echo "WORKER slot=${slot_idx} gpu=${gpu_id} pick index=$((idx+1)) object_id=${object_id}" | tee -a "${worker_log}"
    process_object "${object_id}" "$((idx+1))" "${gpu_id}" 2>&1 | tee -a "${worker_log}"
  done
  echo "WORKER slot=${slot_idx} gpu=${gpu_id} done" | tee -a "${worker_log}"
}

PIDS=()
for ((slot_idx=0; slot_idx<TOTAL_SLOTS; slot_idx++)); do
  worker_main "${slot_idx}" "${SLOT_GPU_IDS[slot_idx]}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "All requested object ids processed."
echo "Done list: ${DONE_LIST}"
echo "Failed list: ${FAILED_LIST}"
echo "Caption failed list: ${CAPTION_FAILED_LIST}"
echo "Generated samples:"
find "${OUTPUT_ROOT}/train" -type f -name metadata.json 2>/dev/null | wc -l

exit "${status}"

# bash /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/run_try1_physxnet_all_objects_all_cases0419.sh
# GPU_IDS=5,6,7 PROCS_PER_GPU=2 bash /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/run_try1_physxnet_all_objects_all_cases0419.sh
