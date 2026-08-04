#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/pck_extreme_benchmark_test5_ready"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_pck_all_conditions_bench.py"

BENCH_RUN_METRICS=1 \
BENCH_INPUT_JSON_ALLOWLIST="${OUTPUT_ROOT}/all_conditions_input_json_allowlist.txt" \
BENCH_RESULT_DIR="${OUTPUT_ROOT}/summaries_all_conditions" \
bash "${SCRIPT_DIR}/bench.sh" "${OUTPUT_ROOT}/bench_new_conditions_methods.txt"
