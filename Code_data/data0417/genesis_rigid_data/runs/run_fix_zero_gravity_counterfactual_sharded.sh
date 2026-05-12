#!/usr/bin/env bash
# 用途：分片修复零重力 counterfactual 样本。
set -euo pipefail

WAN_PY="${WAN_PY:-/data/gaoya/miniconda3/envs/wan/bin/python}"
PY3_BIN="${PY3_BIN:-python3}"
SCRIPT="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generators/try1_physxnet_articulation_mpm0417.py"
TASK_SCRIPT="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/repair/list_zero_gravity_counterfactual_tasks.py"
PHYSX_ROOT="${PHYSX_ROOT:-/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet}"
VERSION="${VERSION:-version_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases}"

GPU_ID="${GPU_ID:-0}"
PROCS_PER_GPU="${PROCS_PER_GPU:-15}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-4}"
REBUILD_TASKS="${REBUILD_TASKS:-0}"

FIX_ROOT="${FIX_ROOT:-/home/gaoya/Code_Video/Code_data/Code_train/train_0419/zero_gravity_counterfactual_fix}"
TASKS_TSV="${FIX_ROOT}/zero_gravity_tasks.tsv"
SUMMARY_JSON="${FIX_ROOT}/zero_gravity_tasks_summary.json"
RUN_TAG="gpu${GPU_ID}_shard${SHARD_INDEX}of${SHARD_COUNT}"
WORK_ROOT="${FIX_ROOT}/runs/${RUN_TAG}"
WORKER_LOG_DIR="${WORK_ROOT}/workers"
FAILED_LIST="${WORK_ROOT}/failed_tasks.tsv"
DONE_LIST="${WORK_ROOT}/done_tasks.tsv"

mkdir -p "${FIX_ROOT}" "${WORK_ROOT}" "${WORKER_LOG_DIR}"
: > "${FAILED_LIST}"
: > "${DONE_LIST}"

if [[ ! -x "${WAN_PY}" ]]; then
  echo "ERROR: wan python not found: ${WAN_PY}" >&2
  exit 1
fi
if ! command -v "${PY3_BIN}" >/dev/null 2>&1; then
  echo "ERROR: python3 not found: ${PY3_BIN}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: generator script not found: ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${TASK_SCRIPT}" ]]; then
  echo "ERROR: task script not found: ${TASK_SCRIPT}" >&2
  exit 1
fi

if [[ "${REBUILD_TASKS}" == "1" || ! -f "${TASKS_TSV}" ]]; then
  "${PY3_BIN}" "${TASK_SCRIPT}" \
    --output_root "${OUTPUT_ROOT}" \
    --tasks_tsv "${TASKS_TSV}" \
    --summary_json "${SUMMARY_JSON}"
fi

if [[ ! -f "${TASKS_TSV}" ]]; then
  echo "ERROR: tasks file missing after build: ${TASKS_TSV}" >&2
  exit 1
fi

mapfile -t ALL_TASKS < <(tail -n +2 "${TASKS_TSV}")
TOTAL_ALL="${#ALL_TASKS[@]}"
TASKS=()
for ((global_idx=0; global_idx<TOTAL_ALL; global_idx++)); do
  if (( global_idx % SHARD_COUNT == SHARD_INDEX )); then
    TASKS+=("${ALL_TASKS[global_idx]}")
  fi
done
TOTAL="${#TASKS[@]}"

echo "WAN_PY=${WAN_PY}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "GPU_ID=${GPU_ID}"
echo "PROCS_PER_GPU=${PROCS_PER_GPU}"
echo "SHARD_INDEX=${SHARD_INDEX}"
echo "SHARD_COUNT=${SHARD_COUNT}"
echo "TOTAL_ALL=${TOTAL_ALL}"
echo "TOTAL_THIS_SHARD=${TOTAL}"
echo "TASKS_TSV=${TASKS_TSV}"
echo "WORK_ROOT=${WORK_ROOT}"

process_task() {
  local task_line="$1"
  local idx="$2"
  IFS=$'\t' read -r object_id scene_composition count_bucket target_count parent_case_index parent_case_name example_case_name counterfactual_kind example_scene_input <<< "${task_line}"
  local zero_case_dir
  zero_case_dir="$(dirname "${example_scene_input}")"

  local safe_case_name
  safe_case_name="$(echo "${object_id}__${parent_case_name}" | tr '/' '_')"
  local log_path="${WORK_ROOT}/${safe_case_name}.log"
  : > "${log_path}"

  echo "==> [${idx}/${TOTAL}] object_id=${object_id} bucket=${scene_composition}/${count_bucket} parent=${parent_case_name} gpu=${GPU_ID}" | tee -a "${log_path}"
  echo "TASK example_case=${example_case_name} kind=${counterfactual_kind}" | tee -a "${log_path}"
  echo "TASK source=${example_scene_input}" | tee -a "${log_path}"
  if [[ "${counterfactual_kind}" == "no_collision_negative" && -d "${zero_case_dir}" ]]; then
    echo "REMOVE stale_zero_gravity_dir=${zero_case_dir}" | tee -a "${log_path}"
    rm -rf "${zero_case_dir}"
  fi

  set +e
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
    --case_index_filter "${parent_case_index}" \
    --rigid_count_filter "${target_count}" \
    --prefer_existing_runtime_meshes \
    --enable_counterfactual_cases \
    --counterfactual_only \
    --counterfactual_no_collision_gravity_z -9.81 \
    --dt 0.003 \
    --substeps 40 \
    --ball_posx 0.03 \
    --steps 12 \
    --fps 12 \
    --simulator_mode rigid \
    2>&1 | tee -a "${log_path}"
  local status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "FAILED object_id=${object_id} parent=${parent_case_name} gpu=${GPU_ID}" | tee -a "${log_path}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${object_id}" "${scene_composition}" "${count_bucket}" "${target_count}" "${parent_case_index}" "${parent_case_name}" >> "${FAILED_LIST}"
    return 0
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${object_id}" "${scene_composition}" "${count_bucket}" "${target_count}" "${parent_case_index}" "${parent_case_name}" >> "${DONE_LIST}"
}

worker_main() {
  local slot_idx="$1"
  local worker_log="${WORKER_LOG_DIR}/worker_${slot_idx}.log"
  : > "${worker_log}"
  echo "WORKER slot=${slot_idx} gpu=${GPU_ID} start" | tee -a "${worker_log}"
  local idx
  for ((idx=slot_idx; idx<TOTAL; idx+=PROCS_PER_GPU)); do
    local task_line="${TASKS[idx]}"
    echo "WORKER slot=${slot_idx} pick index=$((idx+1))" | tee -a "${worker_log}"
    process_task "${task_line}" "$((idx+1))" 2>&1 | tee -a "${worker_log}"
  done
  echo "WORKER slot=${slot_idx} gpu=${GPU_ID} done" | tee -a "${worker_log}"
}

if [[ "${TOTAL}" -eq 0 ]]; then
  echo "No zero-gravity tasks assigned to this shard."
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
