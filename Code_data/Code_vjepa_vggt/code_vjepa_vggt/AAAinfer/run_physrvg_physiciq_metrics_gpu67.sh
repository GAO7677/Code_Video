#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 PHASELOCK_OFF PHASELOCK_ON WMREWARD_OFF WMREWARD_ON" >&2
  exit 2
fi

PYTHON_BIN="/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python"
BENCH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py"
ALLOWLIST="/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
PYTHONPATH_VALUE="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526"
GPU6=6
GPU7=7

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
  videophy2
  cosmos_reason1
)

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
    result_root="$(realpath "${result_root}")"
    for metric in "${CPU_METRICS[@]}"; do
      run_metric "${result_root}" "${metric}" ""
    done
    for metric in "${GPU_METRICS[@]}"; do
      run_metric "${result_root}" "${metric}" "${gpu}"
    done
  done
}

run_root_group "${GPU6}" "$1" "$2" &
pid6=$!
run_root_group "${GPU7}" "$3" "$4" &
pid7=$!

if wait "${pid6}"; then status6=0; else status6=$?; fi
if wait "${pid7}"; then status7=0; else status7=$?; fi
if [[ "${status6}" -ne 0 || "${status7}" -ne 0 ]]; then
  echo "[metric:failed] gpu6_status=${status6} gpu7_status=${status7}" >&2
  exit 1
fi
echo "[metric:all_done] gpu6=${1},${2} gpu7=${3},${4}"
