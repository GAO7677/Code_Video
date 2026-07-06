#!/usr/bin/env bash
set -u

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
  "phyground"
  "cosmos_reason1"
)

FAILED_METRICS=()

run_metric() {
  local metric="$1"
  echo "[bench] start metric=${metric}"
  if "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --result-root "${RESULT_ROOT}"; then
    echo "[bench] done metric=${metric}"
  else
    echo "[bench] failed metric=${metric}" >&2
    FAILED_METRICS+=("${metric}")
  fi
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
if ! "${PYTHON_BIN}" "${REPORT_PY}" --result-root "${RESULT_ROOT}"; then
  echo "[bench] failed metric report rendering" >&2
  FAILED_METRICS+=("render_report")
fi

if ((${#FAILED_METRICS[@]} > 0)); then
  echo "[bench] completed with failures: ${FAILED_METRICS[*]}" >&2
  exit 1
fi

echo "[bench] all metrics completed successfully"
