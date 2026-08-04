#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
OUTPUT_ROOT="${PCK_EXTREME_BENCH_ROOT:-/data/gaoya/agent-data/outputs/pck_extreme_benchmark_test5_ready}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_pck_extreme_bench_ready.py" \
  --output-root "${OUTPUT_ROOT}"

BENCH_RUN_METRICS=1 \
BENCH_INPUT_JSON_ALLOWLIST="${OUTPUT_ROOT}/input_json_allowlist.txt" \
BENCH_RESULT_DIR="${OUTPUT_ROOT}/summaries" \
bash "${SCRIPT_DIR}/bench.sh" "${OUTPUT_ROOT}/bench_methods.txt"
