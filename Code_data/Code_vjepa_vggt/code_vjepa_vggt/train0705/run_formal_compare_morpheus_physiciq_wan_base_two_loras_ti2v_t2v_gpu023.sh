#!/usr/bin/env bash
set -euo pipefail
# CUDA_VISIBLE_DEVICES=7 /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_wan_base_two_loras_ti2v_t2v_gpu023.sh


# Formal batch run for:
# - morpheus_real_world
# - physicIQ
#
# Modes:
# - ti2v
# - t2v
#
# Methods:
# - wan_base
# - openvid_lora_step10000
# - openvid_0613pybullet_lora_step000500
#
# Output layout:
# - /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_{dataset}/{method}
# - /data/gaoya/AAA_test_video/0623/test/t2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_{dataset}/{method}
#
# Execution policy:
# - Use only GPU 0 / 1 / 2
# - Before each launch, detect which candidate GPU is genuinely idle and use that one
# - For each (dataset, mode), run up to 3 methods in parallel across the idle GPUs
# - After each (dataset, mode) finishes, run bench.sh on that mode root
# - Keep per-case logs beside outputs
# - Keep orchestration logs under /data/gaoya/agent-data
#
# Full run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_wan_base_two_loras_ti2v_t2v_gpu023.sh
#
# Optional filters:
# TARGET_DATASETS="physicIQ" TARGET_MODES="ti2v" OVERWRITE=1 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_wan_base_two_loras_ti2v_t2v_gpu023.sh

PY=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
REPO=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
ENTRY="${REPO}/code_vjepa_vggt/AAAinfer/wan_base_two_loras_ti2v_t2v.py"
BATCH_ENTRY="${REPO}/code_vjepa_vggt/AAAinfer/wan_base_two_loras_ti2v_t2v_batch.py"
BENCH_SCRIPT="${REPO}/code_vjepa_vggt/AAAinfer/bench.sh"

LIST_MORPHEUS=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt
LIST_PHYSICIQ=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt

RESULT_BASE_TI2V=/data/gaoya/AAA_test_video/0623/test/ti2v
RESULT_BASE_T2V=/data/gaoya/AAA_test_video/0623/test/t2v
LOG_BASE=/data/gaoya/agent-data/outputs/train0705_wan_base_two_loras_formal_logs_20260706
mkdir -p "${RESULT_BASE_TI2V}" "${RESULT_BASE_T2V}" "${LOG_BASE}"

TARGET_DATASETS="${TARGET_DATASETS:-morpheus_real_world physicIQ}"
TARGET_MODES="${TARGET_MODES:-ti2v t2v}"
OVERWRITE="${OVERWRITE:-0}"
GPU_POOL=(0 1 2)
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
GPU_MAX_MEMORY_MB="${GPU_MAX_MEMORY_MB:-1024}"
GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION:-10}"
FORCE_GPU="${FORCE_GPU:-}"

FAILED_JOBS=()
LAST_LAUNCHED_PID=""
declare -A GPU_PID=()
declare -A GPU_LABEL=()

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 127
fi

dataset_list_path() {
  local dataset_tag="$1"
  case "${dataset_tag}" in
    morpheus_real_world) echo "${LIST_MORPHEUS}" ;;
    physicIQ) echo "${LIST_PHYSICIQ}" ;;
    *)
      echo "unknown dataset_tag: ${dataset_tag}" >&2
      return 1
      ;;
  esac
}

mode_result_root() {
  local mode="$1"
  local dataset_tag="$2"
  case "${mode}" in
    ti2v) echo "${RESULT_BASE_TI2V}/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_${dataset_tag}" ;;
    t2v) echo "${RESULT_BASE_T2V}/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_${dataset_tag}" ;;
    *)
      echo "unknown mode: ${mode}" >&2
      return 1
      ;;
  esac
}

gpu_is_available() {
  local gpu="$1"
  local row
  row="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${row}" ]]; then
    return 1
  fi

  local index
  local memory_used
  local utilization
  IFS=',' read -r index memory_used utilization <<< "${row}"
  memory_used="$(echo "${memory_used}" | xargs)"
  utilization="$(echo "${utilization}" | xargs)"

  local app_count
  app_count="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
      | sed '/^[[:space:]]*$/d' \
      | wc -l
  )"

  if (( app_count > 0 )); then
    return 1
  fi
  if (( memory_used > GPU_MAX_MEMORY_MB )); then
    return 1
  fi
  if (( utilization > GPU_MAX_UTILIZATION )); then
    return 1
  fi
  return 0
}

describe_gpu() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
    | head -n 1 \
    | sed 's/^[[:space:]]*//'
}

check_freed_gpus() {
  local gpu
  for gpu in "${!GPU_PID[@]}"; do
    local pid="${GPU_PID[$gpu]:-}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      unset 'GPU_PID[$gpu]'
      unset 'GPU_LABEL[$gpu]'
    fi
  done
}

wait_for_free_gpu() {
  if [[ -n "${FORCE_GPU}" ]]; then
    while :; do
      check_freed_gpus
      if [[ -z "${GPU_PID[$FORCE_GPU]:-}" ]]; then
        echo "${FORCE_GPU}"
        return 0
      fi
      echo "[gpu_wait] forced gpu=${FORCE_GPU} still occupied by ${GPU_LABEL[$FORCE_GPU]:-unknown}; poll=${GPU_POLL_SECONDS}s" >&2
      sleep "${GPU_POLL_SECONDS}"
    done
  fi

  while :; do
    check_freed_gpus

    local gpu
    for gpu in "${GPU_POOL[@]}"; do
      if [[ -n "${GPU_PID[$gpu]:-}" ]]; then
        continue
      fi
      if gpu_is_available "${gpu}"; then
        echo "${gpu}"
        return 0
      fi
    done

    echo "[gpu_wait] no idle gpu in pool ${GPU_POOL[*]}; poll=${GPU_POLL_SECONDS}s thresholds memory<=${GPU_MAX_MEMORY_MB}MB util<=${GPU_MAX_UTILIZATION}%" >&2
    for gpu in "${GPU_POOL[@]}"; do
      local owner="${GPU_LABEL[$gpu]:-(external_or_busy)}"
      echo "[gpu_wait] gpu=${gpu} owner=${owner} status=$(describe_gpu "${gpu}")" >&2
    done
    sleep "${GPU_POLL_SECONDS}"
  done
}

launch_method_job() {
  local mode="$1"
  local dataset_tag="$2"
  local list_path="$3"
  local method_name="$4"
  local dataset_root="$5"

  local method_root="${dataset_root}/${method_name}"
  local job_label="${dataset_tag}:${mode}:${method_name}"
  local job_log="${LOG_BASE}/${dataset_tag}_${mode}_${method_name}.log"
  local gpu
  mkdir -p "${method_root}"
  gpu="$(wait_for_free_gpu)"

  echo "[launch] label=${job_label} gpu=${gpu} method_root=${method_root}" >&2

  (
    export PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PY}" -u "${BATCH_ENTRY}" \
      --mode "${mode}" \
      --model-preset "${method_name}" \
      --input-json-list-path "${list_path}" \
      --output-root "${method_root}" \
      --num-inference-steps 40 \
      --cfg-scale 5.0 \
      --seed 42 \
      --fps 30 \
      --num-frames 24 \
      $( [[ "${OVERWRITE}" == "1" ]] && printf -- '--overwrite' )
  ) >"${job_log}" 2>&1 &

  local pid=$!
  GPU_PID["${gpu}"]="${pid}"
  GPU_LABEL["${gpu}"]="${job_label}"
  LAST_LAUNCHED_PID="${pid}"
}

wait_method_jobs() {
  local dataset_tag="$1"
  local mode="$2"
  shift 2
  local pids=("$@")
  local pid

  for pid in "${pids[@]}"; do
    if wait "${pid}"; then
      echo "[job:done] dataset=${dataset_tag} mode=${mode} pid=${pid}"
    else
      local rc=$?
      echo "[job:failed] dataset=${dataset_tag} mode=${mode} pid=${pid} rc=${rc}" >&2
      FAILED_JOBS+=("${dataset_tag}:${mode}:pid${pid}:rc${rc}")
    fi

    local gpu
    for gpu in "${!GPU_PID[@]}"; do
      if [[ "${GPU_PID[$gpu]:-}" == "${pid}" ]]; then
        unset 'GPU_PID[$gpu]'
        unset 'GPU_LABEL[$gpu]'
      fi
    done
  done
}

run_dataset_mode() {
  local dataset_tag="$1"
  local mode="$2"
  local list_path
  list_path="$(dataset_list_path "${dataset_tag}")"
  local dataset_root
  dataset_root="$(mode_result_root "${mode}" "${dataset_tag}")"
  mkdir -p "${dataset_root}"

  echo "============================================================"
  echo "[dataset_mode:start] dataset=${dataset_tag} mode=${mode}"
  echo "[dataset_mode:list]  ${list_path}"
  echo "[dataset_mode:root]  ${dataset_root}"
  echo "============================================================"

  if [[ -n "${FORCE_GPU}" ]]; then
    local method_name
    local method_pid
    for method_name in "wan_base" "openvid_lora_step10000" "openvid_0613pybullet_lora_step000500"; do
      launch_method_job "${mode}" "${dataset_tag}" "${list_path}" "${method_name}" "${dataset_root}"
      method_pid="${LAST_LAUNCHED_PID}"
      wait_method_jobs "${dataset_tag}" "${mode}" "${method_pid}"
    done
  else
    local pid_wan_base
    local pid_openvid
    local pid_pybullet
    launch_method_job "${mode}" "${dataset_tag}" "${list_path}" "wan_base" "${dataset_root}"
    pid_wan_base="${LAST_LAUNCHED_PID}"
    launch_method_job "${mode}" "${dataset_tag}" "${list_path}" "openvid_lora_step10000" "${dataset_root}"
    pid_openvid="${LAST_LAUNCHED_PID}"
    launch_method_job "${mode}" "${dataset_tag}" "${list_path}" "openvid_0613pybullet_lora_step000500" "${dataset_root}"
    pid_pybullet="${LAST_LAUNCHED_PID}"

    wait_method_jobs "${dataset_tag}" "${mode}" "${pid_wan_base}" "${pid_openvid}" "${pid_pybullet}"
  fi

  if ((${#FAILED_JOBS[@]} > 0)); then
    echo "[dataset_mode:warning] some jobs failed before bench: ${FAILED_JOBS[*]}" >&2
  fi

  local bench_gpu
  bench_gpu="$(wait_for_free_gpu)"
  echo "[bench:start] dataset=${dataset_tag} mode=${mode} gpu=${bench_gpu}"
  CUDA_VISIBLE_DEVICES="${bench_gpu}" BENCH_CUDA_VISIBLE_DEVICES="${bench_gpu}" bash "${BENCH_SCRIPT}" "${dataset_root}"
  echo "[bench:done] dataset=${dataset_tag} mode=${mode} gpu=${bench_gpu}"
}

for dataset_tag in ${TARGET_DATASETS}; do
  for mode in ${TARGET_MODES}; do
    run_dataset_mode "${dataset_tag}" "${mode}"
  done
done

if ((${#FAILED_JOBS[@]} > 0)); then
  echo "[all_done_with_errors] ${FAILED_JOBS[*]}" >&2
  exit 1
fi

echo "[all_done] wan_base + two loras formal comparison finished successfully"
