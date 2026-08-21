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
MIN_FREE_MIB=22000

HEAVY_METRICS=(
  videophy2
  cosmos_reason1
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
  local gpu="$3"
  local summary="${result_root}/eval_summary_${metric}.json"
  wait_for_gpu_memory "$gpu"
  echo "[metric:start] root=${result_root} metric=${metric} cuda=${gpu}"
  env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PYTHONPATH_VALUE}" \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${BENCH}" \
      --metric "${metric}" \
      --result-root "${result_root}" \
      --input-json-allowlist "${ALLOWLIST}" \
      --output-summary "${summary}"
  echo "[metric:done] root=${result_root} metric=${metric} cuda=${gpu}"
}

run_root_group() {
  local gpu="$1"
  shift
  for result_root in "$@"; do
    result_root="$(realpath "$result_root")"
    wait_for_complete_root "$result_root"
    for metric in "${HEAVY_METRICS[@]}"; do
      run_metric "$result_root" "$metric" "$gpu"
    done
  done
}

# One heavy metric process per physical GPU. GPU0 handles PhaseLock OFF and
# then WMReward ON; GPU2 handles PhaseLock ON; GPU3 waits for WMReward OFF.
# No GPU4 is referenced, and VideoPhy2/Cosmos are never concurrent on a card.
run_root_group 0 "$1" "$4" &
pid0=$!
run_root_group 2 "$2" &
pid2=$!
run_root_group 3 "$3" &
pid3=$!

if wait "$pid0"; then status0=0; else status0=$?; fi
if wait "$pid2"; then status2=0; else status2=$?; fi
if wait "$pid3"; then status3=0; else status3=$?; fi
if [[ "$status0" -ne 0 || "$status2" -ne 0 || "$status3" -ne 0 ]]; then
  echo "[metric:failed] gpu0_status=${status0} gpu2_status=${status2} gpu3_status=${status3}" >&2
  exit 1
fi
echo "[metric:all_done] gpu0=$1,$4 gpu2=$2 gpu3=$3"
