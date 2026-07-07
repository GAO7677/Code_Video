#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRY0526_ROOT="/home/gaoya/Code_Video/Code_data/Code_try0526"

PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
BENCH_PY="${SCRIPT_DIR}/bench.py"
REPORT_PY="${SCRIPT_DIR}/render_v2v_metric_report.py"

RESULT_ROOT="${1:-/data/gaoya/AAA_test_video/0623/test/v2v}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"

export PYTHONPATH="${PROJECT_ROOT}:${TRY0526_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${BENCH_CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES}"
fi

METRICS=(
  "wmreward"
  "physics_iq"
  "physics_iq_with_context"
  "physics_iq_without_context"
  "pmf_with_context"
  "pmf_without_context"
  "videophy2"
  # "phyground"
  "cosmos_reason1"
)

run_metric() {
  local metric="$1"
  echo "[bench] start metric=${metric}"
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --result-root "${RESULT_ROOT}"
  echo "[bench] done metric=${metric}"
}

echo "[bench] python=${PYTHON_BIN}"
echo "[bench] result_root=${RESULT_ROOT}"
echo "[bench] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[bench] input_json_policy=read absolute input_json directly from each result json"
echo "[bench] skip_policy=existing metric fields are preserved unless --overwrite is used in bench.py"

for metric in "${METRICS[@]}"; do
  run_metric "${metric}"
done

echo "[bench] render report"
"${PYTHON_BIN}" "${REPORT_PY}" --result-root "${RESULT_ROOT}"

echo "[bench] all metrics completed successfully"
