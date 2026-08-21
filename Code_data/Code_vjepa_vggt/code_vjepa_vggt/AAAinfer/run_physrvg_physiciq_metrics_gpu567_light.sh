#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 PHASELOCK_OFF PHASELOCK_ON WMREWARD_OFF WMREWARD_ON" >&2
  exit 2
fi

PYTHON_BIN="/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python"
BENCH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py"
ALLOWLIST="/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
PYTHONPATH_VALUE="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_try0526"
EXPECTED_CASES=67
MIN_FREE_MIB=16000

CPU_METRICS=(
  physics_iq_with_context
  physics_iq_without_context
  pmf_with_context
  pmf_without_context
)
GPU_METRICS=(
  wmreward
  vbench_subject_consistency
  vbench_background_consistency
  vbench_temporal_flickering
  vbench_motion_smoothness
  vbench_dynamic_degree
  vbench_aesthetic_quality
  vbench_imaging_quality
)

case_count() {
  local result_root="$1"
  find "$result_root" -maxdepth 1 -type f -name '*.mp4' -printf '.' 2>/dev/null | wc -c
}

wait_for_complete_root() {
  local result_root="$1"
  while true; do
    local videos
    videos="$(case_count "$result_root")"
    if [[ "$videos" -eq "$EXPECTED_CASES" ]]; then
      echo "[root:ready] root=${result_root} videos=${videos}/${EXPECTED_CASES}"
      return 0
    fi
    echo "[root:wait] root=${result_root} videos=${videos}/${EXPECTED_CASES}"
    sleep 30
  done
}

gpu_free_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits \
    | tr -d ' ' | head -n 1
}

wait_for_gpu_memory() {
  local gpu="$1"
  while true; do
    local free_mib
    free_mib="$(gpu_free_mib "$gpu")"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      echo "[gpu:ready] gpu=${gpu} free_mib=${free_mib} threshold=${MIN_FREE_MIB}"
      return 0
    fi
    echo "[gpu:wait] gpu=${gpu} free_mib=${free_mib:-unknown} threshold=${MIN_FREE_MIB}"
    sleep 30
  done
}

run_metric() {
  local result_root="$1"
  local metric="$2"
  local visible_devices="$3"
  local summary="${result_root}/eval_summary_${metric}.json"
  echo "[metric:start] root=${result_root} metric=${metric} cuda=${visible_devices:-cpu}"
  env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PYTHONPATH_VALUE}" \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
    "${PYTHON_BIN}" "${BENCH}" \
      --metric "${metric}" \
      --result-root "${result_root}" \
      --input-json-allowlist "${ALLOWLIST}" \
      --output-summary "${summary}" \
      --wmreward-reset-interval 1000000
  echo "[metric:done] root=${result_root} metric=${metric}"
}

run_root_group() {
  local gpu="$1"
  shift
  for result_root in "$@"; do
    result_root="$(realpath "$result_root")"
    wait_for_complete_root "$result_root"
    for metric in "${CPU_METRICS[@]}"; do
      run_metric "$result_root" "$metric" ""
    done
    for metric in "${GPU_METRICS[@]}"; do
      wait_for_gpu_memory "$gpu"
      run_metric "$result_root" "$metric" "$gpu"
    done
  done
}

# GPU5 handles PhaseLock OFF then waits for WMReward ON; GPU6 handles PhaseLock
# ON; GPU7 waits for WMReward OFF. Each physical GPU has at most one metric
# process, and no VideoPhy2/Cosmos process is launched here.
run_root_group 5 "$1" "$4" &
pid5=$!
run_root_group 6 "$2" &
pid6=$!
run_root_group 7 "$3" &
pid7=$!

if wait "$pid5"; then status5=0; else status5=$?; fi
if wait "$pid6"; then status6=0; else status6=$?; fi
if wait "$pid7"; then status7=0; else status7=$?; fi
if [[ "$status5" -ne 0 || "$status6" -ne 0 || "$status7" -ne 0 ]]; then
  echo "[metric:failed] gpu5_status=${status5} gpu6_status=${status6} gpu7_status=${status7}" >&2
  exit 1
fi
echo "[metric:all_done] gpu5=$1,$4 gpu6=$2 gpu7=$3"
