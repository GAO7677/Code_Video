#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRY0526_ROOT="/home/gaoya/Code_Video/Code_data/Code_try0526"

PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
BENCH_PY="${SCRIPT_DIR}/bench.py"
BASELINE_LIST="${1:-${SCRIPT_DIR}/baseline.txt}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
BENCH_METRICS_RAW="${BENCH_METRICS:-}"

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
  "cosmos_reason1"
)

if [[ -n "${BENCH_METRICS_RAW}" ]]; then
  IFS=',' read -r -a METRICS <<< "${BENCH_METRICS_RAW}"
fi

echo "[baseline-bench] python=${PYTHON_BIN}"
echo "[baseline-bench] baseline_list=${BASELINE_LIST}"
echo "[baseline-bench] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[baseline-bench] metrics=${METRICS[*]}"

for metric in "${METRICS[@]}"; do
  echo "[baseline-bench] start metric=${metric}"
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --baseline-list "${BASELINE_LIST}"
  echo "[baseline-bench] done metric=${metric}"
done

echo "[baseline-bench] all metrics completed"
