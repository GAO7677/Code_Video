#!/usr/bin/env bash
set -euo pipefail

# Usage:
# CUDA_VISIBLE_DEVICES=0 \
# LIMIT_PER_FOLDER=1 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_bench_ti2v_t2v.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
TRY0526_ROOT="/home/gaoya/Code_Video/Code_data/Code_try0526"

PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
PREPARE_PY="${SCRIPT_DIR}/prepare_ti2v_t2v_bench_inputs.py"
BENCH_SH="${PACKAGE_ROOT}/AAAinfer/bench.sh"

SOURCE_TI2V="${SOURCE_TI2V:-/data/gaoya/AAA_test_video/0623/test/ti2v}"
SOURCE_T2V="${SOURCE_T2V:-/data/gaoya/AAA_test_video/0623/test/t2v}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/train0705_ti2v_t2v_bench_inputs}"
LIMIT_PER_FOLDER="${LIMIT_PER_FOLDER:-}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
TI2V_T2V_BENCH_METRICS="${TI2V_T2V_BENCH_METRICS:-wmreward,physics_iq,physics_iq_with_context,pmf_with_context,videophy2,cosmos_reason1}"

export PYTHONPATH="${PROJECT_ROOT}:${TRY0526_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${BENCH_CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES}"
fi

prepare_cmd=(
  "${PYTHON_BIN}"
  "${PREPARE_PY}"
  --source-root "${SOURCE_TI2V}"
  --source-root "${SOURCE_T2V}"
  --output-root "${OUTPUT_ROOT}"
  --overwrite
)
if [[ -n "${LIMIT_PER_FOLDER}" ]]; then
  prepare_cmd+=(--limit-per-folder "${LIMIT_PER_FOLDER}")
fi

echo "[ti2v_t2v_bench] prepare output_root=${OUTPUT_ROOT}"
"${prepare_cmd[@]}"
echo "[ti2v_t2v_bench] metrics=${TI2V_T2V_BENCH_METRICS}"

overall_exit_code=0
overall_signal="success"

for mode in ti2v t2v; do
  mode_root="${OUTPUT_ROOT}/${mode}"
  if [[ ! -d "${mode_root}" ]]; then
    echo "[ti2v_t2v_bench] mode=${mode} signal=missing_root root=${mode_root}"
    overall_exit_code=2
    overall_signal="failed"
    continue
  fi

  echo "[ti2v_t2v_bench] mode=${mode} start root=${mode_root}"
  set +e
  BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES}" \
  BENCH_METRICS="${TI2V_T2V_BENCH_METRICS}" \
  bash "${BENCH_SH}" "${mode_root}"
  mode_exit_code=$?
  set -e
  if [[ "${mode_exit_code}" -eq 0 ]]; then
    mode_signal="success"
  else
    mode_signal="failed"
    overall_exit_code="${mode_exit_code}"
    overall_signal="failed"
  fi
  echo "[ti2v_t2v_bench] mode=${mode} signal=${mode_signal} exit_code=${mode_exit_code} root=${mode_root}"
done

echo "[ti2v_t2v_bench] final_signal=${overall_signal} exit_code=${overall_exit_code} output_root=${OUTPUT_ROOT}"
exit "${overall_exit_code}"
