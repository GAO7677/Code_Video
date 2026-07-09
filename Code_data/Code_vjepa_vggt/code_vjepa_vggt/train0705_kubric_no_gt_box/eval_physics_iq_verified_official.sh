#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYSIQ_REPO="${PHYSIQ_REPO:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/physics-IQ-benchmark-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"

RUN_ROOT="${RUN_ROOT:-}"
RUN_NAME="${RUN_NAME:-}"
INPUT_FOLDER="${INPUT_FOLDER:-}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-/data/gaoya/agent-data/outputs/physics_iq_verified_eval}"
BENCHMARK_BASE_FOLDER="${BENCHMARK_BASE_FOLDER:-/data/gaoya/dataset}"
DESCRIPTIONS_FILE="${DESCRIPTIONS_FILE:-${PHYSIQ_REPO}/descriptions/best_practice/descriptions_base.csv}"
AGGREGATE_SCORE="${AGGREGATE_SCORE:-1}"
RUN_GLOB="${RUN_GLOB:-*-bpp-run_*}"

if [[ ! -d "${PHYSIQ_REPO}" ]]; then
  echo "[error] physics-IQ repo not found: ${PHYSIQ_REPO}" >&2
  exit 1
fi

if [[ -n "${INPUT_FOLDER}" ]]; then
  INPUTS=("${INPUT_FOLDER}")
elif [[ -n "${RUN_ROOT}" ]]; then
  mapfile -t INPUTS < <(find "${RUN_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "${RUN_GLOB}" | sort)
elif [[ -n "${RUN_NAME}" ]]; then
  INPUTS=("${OUTPUT_FOLDER}/${RUN_NAME}")
else
  echo "[error] set INPUT_FOLDER or RUN_ROOT" >&2
  exit 1
fi

if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  echo "[error] no input run folders resolved" >&2
  exit 1
fi

echo "[eval] input folders:"
printf '  %s\n' "${INPUTS[@]}"

cd "${PHYSIQ_REPO}"

EVAL_CMD=(
  "${PYTHON_BIN}"
  physiq/run_physics_iq.py
  --input_folders
)
for folder in "${INPUTS[@]}"; do
  EVAL_CMD+=("${folder}")
done
EVAL_CMD+=(
  --output_folder "${OUTPUT_FOLDER}"
  --descriptions_file "${DESCRIPTIONS_FILE}"
  --benchmark_base_folder "${BENCHMARK_BASE_FOLDER}"
)

echo "[eval] ${EVAL_CMD[*]}"
"${EVAL_CMD[@]}"

if [[ "${AGGREGATE_SCORE}" != "1" ]]; then
  exit 0
fi

RESULTS_DIR="${OUTPUT_FOLDER}/physics-IQ-benchmark-verified/results"
CSV_FILES=()
for folder in "${INPUTS[@]}"; do
  stem="$(basename "${folder}")"
  csv_path="${RESULTS_DIR}/${stem}.csv"
  if [[ ! -f "${csv_path}" ]]; then
    echo "[warn] missing csv for aggregation: ${csv_path}" >&2
    continue
  fi
  CSV_FILES+=("${csv_path}")
done

if [[ "${#CSV_FILES[@]}" -eq 0 ]]; then
  echo "[warn] no csv files found for aggregation under ${RESULTS_DIR}" >&2
  exit 0
fi

AGG_CMD=(
  "${PYTHON_BIN}"
  physiq/aggregate_runs_from_csvs.py
)
for csv_path in "${CSV_FILES[@]}"; do
  AGG_CMD+=("${csv_path}")
done
AGG_CMD+=(
  --score-type verified
)

echo "[aggregate] ${AGG_CMD[*]}"
"${AGG_CMD[@]}"
