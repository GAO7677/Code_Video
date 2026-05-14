#!/usr/bin/env bash
# 用途：安全批量生成 100 条 generic-v22 样本；同一 object_id 不共享同一输出根目录，降低 Genesis 并发冲突。
set -euo pipefail

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan

SCRIPT="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generators/try1_physxnet_articulation_mpm0417.py"
FINAL_ROOT="${FINAL_ROOT:-/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_genericv22_batch100}"
SHARD_ROOT="${SHARD_ROOT:-${FINAL_ROOT}_shards}"
LOG_ROOT="${FINAL_ROOT}/_logs"
PHYSX_ROOT="/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet"
VERSION="version_1"

GPU_19925_ENV="${GPU_19925_ENV:-3}"
GPU_19925_MULTI="${GPU_19925_MULTI:-4}"
GPU_10037="${GPU_10037:-5}"
GPU_5050="${GPU_5050:-6}"
VIDEO_SLOWMO_PROB="${VIDEO_SLOWMO_PROB:-0.35}"
VIDEO_SLOWMO_FACTORS=(${VIDEO_SLOWMO_FACTORS:-1.15 1.30})

mkdir -p "${FINAL_ROOT}" "${SHARD_ROOT}" "${LOG_ROOT}"
: > "${LOG_ROOT}/done_tasks.txt"
: > "${LOG_ROOT}/failed_tasks.txt"

case_index_to_name() {
  local case_idx="$1"
  case "${case_idx}" in
    5) echo "case005_entry_left" ;;
    210) echo "case210_multi2_projectile_nocollision" ;;
    211) echo "case211_multi2_drop_nocollision" ;;
    220) echo "case220_multi3_projectile_nocollision" ;;
    221) echo "case221_multi3_drop_nocollision" ;;
    *) return 1 ;;
  esac
}

sample_dir_exists() {
  local output_root="$1"
  local object_id="$2"
  local case_name="$3"
  local rs="$4"
  local rs_tag
  rs_tag="$(printf '__rs%02d' "${rs}")"
  find "${output_root}/train/rigid" -type d -name "${object_id}__${case_name}${rs_tag}" -print -quit 2>/dev/null | grep -q .
}

run_one() {
  local output_root="$1"
  local gpu_id="$2"
  local object_id="$3"
  local target_count="$4"
  local case_index="$5"
  local rs="$6"
  local leaf_bucket="$7"
  local shard_name="$8"
  local case_name
  case_name="$(case_index_to_name "${case_index}")"
  local worker_log="${LOG_ROOT}/${shard_name}_${object_id}_c${case_index}_rs${rs}.log"
  echo "START shard=${shard_name} gpu=${gpu_id} object=${object_id} count=${target_count} case=${case_index} rs=${rs} leaf=${leaf_bucket}" | tee -a "${worker_log}"
  if CUDA_VISIBLE_DEVICES="${gpu_id}" python3 "${SCRIPT}" \
      --physx_root "${PHYSX_ROOT}" \
      --version "${VERSION}" \
      --object_id "${object_id}" \
      --output_root "${output_root}" \
      --run_genesis \
      --generate_all_count_motion_cases \
      --rigid_count_filter "${target_count}" \
      --case_index_filter "${case_index}" \
      --simple_case_resample_index "${rs}" \
      --prefer_existing_runtime_meshes \
      --dt 0.003 \
      --substeps 40 \
      --steps 49 \
      --fps 12 \
      --video_slowmo_prob "${VIDEO_SLOWMO_PROB}" \
      --video_slowmo_factors "${VIDEO_SLOWMO_FACTORS[@]}" \
      --simulator_mode rigid \
      >> "${worker_log}" 2>&1; then
    if sample_dir_exists "${output_root}" "${object_id}" "${case_name}" "${rs}"; then
      echo -e "${shard_name}\t${object_id}\t${target_count}\t${case_index}\t${rs}\t${leaf_bucket}" >> "${LOG_ROOT}/done_tasks.txt"
    else
      echo -e "${shard_name}\t${object_id}\t${target_count}\t${case_index}\t${rs}\t${leaf_bucket}\tmissing_sample_dir" >> "${LOG_ROOT}/failed_tasks.txt"
    fi
  else
    echo -e "${shard_name}\t${object_id}\t${target_count}\t${case_index}\t${rs}\t${leaf_bucket}\tcommand_failed" >> "${LOG_ROOT}/failed_tasks.txt"
  fi
}

run_series() {
  local shard_name="$1"
  local output_root="$2"
  local gpu_id="$3"
  shift 3
  mkdir -p "${output_root}"
  while [[ "$#" -gt 0 ]]; do
    local object_id="$1"; shift
    local target_count="$1"; shift
    local case_index="$1"; shift
    local rs_begin="$1"; shift
    local rs_end="$1"; shift
    local leaf_bucket="$1"; shift
    local rs
    for ((rs=rs_begin; rs<=rs_end; rs++)); do
      run_one "${output_root}" "${gpu_id}" "${object_id}" "${target_count}" "${case_index}" "${rs}" "${leaf_bucket}" "${shard_name}"
    done
  done
}

merge_shard_train() {
  local shard_root="$1"
  if [[ -d "${shard_root}/train" ]]; then
    mkdir -p "${FINAL_ROOT}/train"
    cp -a "${shard_root}/train/." "${FINAL_ROOT}/train/"
  fi
}

run_series "shard_19925_env" "${SHARD_ROOT}/shard_19925_env" "${GPU_19925_ENV}" \
  "19925" "2" "210" "1" "17" "count_02/env_only" \
  "19925" "2" "211" "1" "17" "count_02/env_only" &
pid_env=$!

run_series "shard_19925_multi" "${SHARD_ROOT}/shard_19925_multi" "${GPU_19925_MULTI}" \
  "19925" "3" "220" "1" "11" "count_03_04/mixed_c2plus" \
  "19925" "3" "221" "1" "11" "count_03_04/mixed_c2plus" &
pid_multi=$!

run_series "shard_10037" "${SHARD_ROOT}/shard_10037" "${GPU_10037}" \
  "10037" "2" "5" "1" "17" "count_02/mixed_c1" \
  "10037" "3" "5" "1" "11" "count_03_04/mixed_c2plus" &
pid_10037=$!

run_series "shard_5050" "${SHARD_ROOT}/shard_5050" "${GPU_5050}" \
  "5050" "2" "210" "1" "16" "count_02/mixed_c1" &
pid_5050=$!

wait "${pid_env}" "${pid_multi}" "${pid_10037}" "${pid_5050}"

rm -rf "${FINAL_ROOT}/train"
mkdir -p "${FINAL_ROOT}/train"
merge_shard_train "${SHARD_ROOT}/shard_19925_env"
merge_shard_train "${SHARD_ROOT}/shard_19925_multi"
merge_shard_train "${SHARD_ROOT}/shard_10037"
merge_shard_train "${SHARD_ROOT}/shard_5050"

echo "MERGED train trees into ${FINAL_ROOT}/train"
echo "done=$(wc -l < "${LOG_ROOT}/done_tasks.txt") failed=$(wc -l < "${LOG_ROOT}/failed_tasks.txt")"
