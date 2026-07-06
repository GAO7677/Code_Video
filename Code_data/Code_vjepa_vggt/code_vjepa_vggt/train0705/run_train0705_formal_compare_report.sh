#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRY0526_ROOT="/home/gaoya/Code_Video/Code_data/Code_try0526"

PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
BENCH_SH="${PROJECT_ROOT}/code_vjepa_vggt/AAAinfer/bench.sh"
REPORT_PY="${PROJECT_ROOT}/code_vjepa_vggt/AAAinfer/render_v2v_metric_report.py"

RESULT_ROOT="${RESULT_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/test/report/v2v/train0705_formal_compare/combined}"
MORPHEUS_LIST="${MORPHEUS_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt}"
PHYSICIQ_LIST="${PHYSICIQ_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"

RUN_BENCH="${RUN_BENCH:-1}"
START_PYPORT="${START_PYPORT:-0}"
PYPORT_PORT="${PYPORT_PORT:-8991}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

export PYTHONPATH="${PROJECT_ROOT}:${TRY0526_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[train0705-report] result_root=${RESULT_ROOT}"
echo "[train0705-report] output_dir=${OUTPUT_DIR}"
echo "[train0705-report] morpheus_list=${MORPHEUS_LIST}"
echo "[train0705-report] physicIQ_list=${PHYSICIQ_LIST}"
echo "[train0705-report] run_bench=${RUN_BENCH}"
echo "[train0705-report] start_pyport=${START_PYPORT}"

if [[ "${RUN_BENCH}" == "1" ]]; then
  echo "[train0705-report] running bench.sh"
  CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES}" \
    bash "${BENCH_SH}" "${RESULT_ROOT}"
fi

echo "[train0705-report] rendering combined report"
"${PYTHON_BIN}" "${REPORT_PY}" \
  --result-root "${RESULT_ROOT}" \
  --input-json-list-path "${MORPHEUS_LIST}" \
  --input-json-list-path "${PHYSICIQ_LIST}" \
  --output-dir "${OUTPUT_DIR}"

echo "[train0705-report] report ready: ${OUTPUT_DIR}/index.html"

if [[ "${START_PYPORT}" == "1" ]]; then
  echo "[train0705-report] starting pyport on port ${PYPORT_PORT}"
  pyport "${OUTPUT_DIR}" "${PYPORT_PORT}"
else
  echo "[train0705-report] preview command: pyport ${OUTPUT_DIR} ${PYPORT_PORT}"
fi
