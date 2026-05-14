#!/usr/bin/env bash
# 用途：批量生成 100 条 generic-v22 相机方案样本，并保持正式 train/rigid 导出格式。
set -euo pipefail

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan

SCRIPT="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generators/try1_physxnet_articulation_mpm0417.py"
PHYSX_ROOT="/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet"
VERSION="version_1"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_genericv22_batch100}"
LOG_DIR="${OUTPUT_ROOT}/_logs"
DONE_LIST="${LOG_DIR}/done_tasks.txt"
FAILED_LIST="${LOG_DIR}/failed_tasks.txt"
GPU_IDS_CSV="${GPU_IDS:-3,4,5,6,7}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
: > "${DONE_LIST}"
: > "${FAILED_LIST}"

IFS=',' read -r -a GPU_IDS_ARR <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_IDS_ARR[@]}" -eq 0 ]]; then
  GPU_IDS_ARR=("0")
fi
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_IDS_ARR[@]}}"

TASK_FILE="${LOG_DIR}/task_plan.tsv"
: > "${TASK_FILE}"

append_tasks() {
  local object_id="$1"
  local target_count="$2"
  local case_index="$3"
  local rs_begin="$4"
  local rs_end="$5"
  local leaf_bucket="$6"
  local template_name="$7"
  local rs
  for ((rs=rs_begin; rs<=rs_end; rs++)); do
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${object_id}" "${target_count}" "${case_index}" "${rs}" "${leaf_bucket}" "${template_name}" >> "${TASK_FILE}"
  done
}

# 100 total:
# - count_02/env_only: 34
# - count_02/mixed_c1: 33
# - count_03_04/mixed_c2plus: 33
append_tasks "19925" "2" "210" "1" "17" "count_02/env_only" "19925_case210"
append_tasks "19925" "2" "211" "1" "17" "count_02/env_only" "19925_case211"
append_tasks "10037" "2" "5"   "1" "17" "count_02/mixed_c1" "10037_case005"
append_tasks "5050"  "2" "210" "1" "16" "count_02/mixed_c1" "5050_case210"
append_tasks "19925" "3" "220" "1" "11" "count_03_04/mixed_c2plus" "19925_case220"
append_tasks "19925" "3" "221" "1" "11" "count_03_04/mixed_c2plus" "19925_case221"
append_tasks "10037" "3" "5"   "1" "11" "count_03_04/mixed_c2plus" "10037_case005"

TOTAL_TASKS="$(wc -l < "${TASK_FILE}")"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "GPU_IDS=${GPU_IDS_CSV}"
echo "MAX_PARALLEL=${MAX_PARALLEL}"
echo "TOTAL_TASKS=${TOTAL_TASKS}"
echo "TASK_FILE=${TASK_FILE}"

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -rp | wc -l | tr -d ' ')"
    if [[ "${running}" -lt "${MAX_PARALLEL}" ]]; then
      break
    fi
    sleep 3
  done
}

launch_task() {
  local gpu_id="$1"
  local object_id="$2"
  local target_count="$3"
  local case_index="$4"
  local resample_index="$5"
  local leaf_bucket="$6"
  local template_name="$7"
  local worker_log="${LOG_DIR}/gpu${gpu_id}_${object_id}_c${case_index}_rs${resample_index}.log"

  (
    set -euo pipefail
    source /home/gaoya/miniconda3/etc/profile.d/conda.sh
    conda activate wan
    echo "START gpu=${gpu_id} object=${object_id} count=${target_count} case=${case_index} rs=${resample_index} leaf=${leaf_bucket} template=${template_name}" | tee -a "${worker_log}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 "${SCRIPT}" \
      --physx_root "${PHYSX_ROOT}" \
      --version "${VERSION}" \
      --object_id "${object_id}" \
      --output_root "${OUTPUT_ROOT}" \
      --run_genesis \
      --generate_all_count_motion_cases \
      --rigid_count_filter "${target_count}" \
      --case_index_filter "${case_index}" \
      --simple_case_resample_index "${resample_index}" \
      --prefer_existing_runtime_meshes \
      --dt 0.003 \
      --substeps 40 \
      --steps 49 \
      --fps 12 \
      --simulator_mode rigid \
      >> "${worker_log}" 2>&1
    echo -e "${object_id}\t${target_count}\t${case_index}\t${resample_index}\t${leaf_bucket}\t${template_name}" >> "${DONE_LIST}"
  ) || {
    echo -e "${object_id}\t${target_count}\t${case_index}\t${resample_index}\t${leaf_bucket}\t${template_name}" >> "${FAILED_LIST}"
    echo "FAILED gpu=${gpu_id} object=${object_id} count=${target_count} case=${case_index} rs=${resample_index}" >&2
  } &
}

task_idx=0
while IFS=$'\t' read -r object_id target_count case_index resample_index leaf_bucket template_name; do
  wait_for_slot
  gpu_id="${GPU_IDS_ARR[$((task_idx % ${#GPU_IDS_ARR[@]}))]}"
  echo "QUEUE idx=$((task_idx + 1))/${TOTAL_TASKS} gpu=${gpu_id} object=${object_id} count=${target_count} case=${case_index} rs=${resample_index} leaf=${leaf_bucket}"
  launch_task "${gpu_id}" "${object_id}" "${target_count}" "${case_index}" "${resample_index}" "${leaf_bucket}" "${template_name}"
  task_idx=$((task_idx + 1))
done < "${TASK_FILE}"

wait
echo "DONE total=${TOTAL_TASKS} done=$(wc -l < "${DONE_LIST}") failed=$(wc -l < "${FAILED_LIST}")"
