#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRY0526_ROOT="/home/gaoya/Code_Video/Code_data/Code_try0526"
DEFAULT_RESULT_ROOT="/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare"
DEFAULT_BASE_RESULT_ROOT="/data/gaoya/AAA_test_video/0623/test/v2v"
DEFAULT_BASE_REPORT_ROOT="/data/gaoya/AAA_test_video/0623/test/report/v2v"

PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
BENCH_SH="${PROJECT_ROOT}/code_vjepa_vggt/AAAinfer/bench.sh"
REPORT_PY="${PROJECT_ROOT}/code_vjepa_vggt/AAAinfer/render_v2v_metric_report.py"

POSITIONAL_RESULT_ROOT="${1:-}"
if [[ -n "${POSITIONAL_RESULT_ROOT}" ]]; then
  shift
fi

RESULT_ROOT="${POSITIONAL_RESULT_ROOT:-${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}}"
MORPHEUS_LIST="${MORPHEUS_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt}"
PHYSICIQ_LIST="${PHYSICIQ_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"

RUN_BENCH="${RUN_BENCH:-1}"
START_PYPORT="${START_PYPORT:-0}"
PYPORT_PORT="${PYPORT_PORT:-8991}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

export PYTHONPATH="${PROJECT_ROOT}:${TRY0526_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

declare -a LIST_ARGS=()
if (($# > 0)); then
  for list_path in "$@"; do
    LIST_ARGS+=("${list_path}")
  done
elif [[ "${RESULT_ROOT}" == "${DEFAULT_RESULT_ROOT}" ]]; then
  LIST_ARGS+=("${MORPHEUS_LIST}" "${PHYSICIQ_LIST}")
fi

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  result_root_resolved="$(realpath -m "${RESULT_ROOT}")"
  if [[ "${result_root_resolved}" == "${DEFAULT_RESULT_ROOT}" ]]; then
    OUTPUT_DIR="${DEFAULT_BASE_REPORT_ROOT}/train0705_formal_compare/combined"
  elif [[ "${result_root_resolved}" == "${DEFAULT_BASE_RESULT_ROOT}" ]]; then
    OUTPUT_DIR="${DEFAULT_BASE_REPORT_ROOT}"
  elif [[ "${result_root_resolved}" == "${DEFAULT_BASE_RESULT_ROOT}/"* ]]; then
    relative_result_dir="${result_root_resolved#${DEFAULT_BASE_RESULT_ROOT}/}"
    OUTPUT_DIR="${DEFAULT_BASE_REPORT_ROOT}/${relative_result_dir}"
  else
    OUTPUT_DIR="${DEFAULT_BASE_REPORT_ROOT}/custom/$(basename "${result_root_resolved}")"
  fi
fi

echo "[train0705-report] result_root=${RESULT_ROOT}"
echo "[train0705-report] output_dir=${OUTPUT_DIR}"
echo "[train0705-report] run_bench=${RUN_BENCH}"
echo "[train0705-report] start_pyport=${START_PYPORT}"
if ((${#LIST_ARGS[@]} > 0)); then
  printf '[train0705-report] input_json_lists=%s\n' "${LIST_ARGS[*]}"
else
  echo "[train0705-report] input_json_lists=<none>"
fi

if [[ "${RUN_BENCH}" == "1" ]]; then
  echo "[train0705-report] running bench.sh"
  CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES}" \
    bash "${BENCH_SH}" "${RESULT_ROOT}"
fi

echo "[train0705-report] rendering report"
report_cmd=(
  "${PYTHON_BIN}" "${REPORT_PY}"
  --result-root "${RESULT_ROOT}"
  --output-dir "${OUTPUT_DIR}"
)
for list_path in "${LIST_ARGS[@]}"; do
  report_cmd+=(--input-json-list-path "${list_path}")
done
"${report_cmd[@]}"

echo "[train0705-report] report ready: ${OUTPUT_DIR}/index.html"

if [[ "${START_PYPORT}" == "1" ]]; then
  echo "[train0705-report] starting pyport on port ${PYPORT_PORT}"
  pyport "${OUTPUT_DIR}" "${PYPORT_PORT}"
else
  echo "[train0705-report] preview command: pyport ${OUTPUT_DIR} ${PYPORT_PORT}"
fi
