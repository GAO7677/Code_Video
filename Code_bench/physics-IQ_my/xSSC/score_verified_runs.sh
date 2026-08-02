#!/usr/bin/env bash
set -euo pipefail

if (($# < 1 || $# > 4)); then
  echo "Usage: score_verified_runs.sh RUN_FOLDER [RUN_FOLDER ... up to 4]" >&2
  exit 2
fi

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MY_ROOT="$(dirname "$SCRIPT_ROOT")"
OFFICIAL_REPO=/home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main
RESULT_BASE=/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified
EVALUATION_ROOT="$RESULT_BASE/evaluation"
DESCRIPTIONS="$OFFICIAL_REPO/descriptions/best_practice/descriptions_base.csv"
export PATH="/home/gaoya/miniconda3/envs/wan-cu128/bin:$PATH"
export UV_CACHE_DIR=/data/gaoya/agent-data/cache/uv
export UV_PROJECT_ENVIRONMENT=/data/gaoya/agent-data/cache/envs/physics-iq-verified

command -v uv >/dev/null 2>&1 || {
  echo "uv is missing; run $SCRIPT_ROOT/setup_official_benchmark_env.sh first" >&2
  exit 1
}

RUN_FOLDERS=("$@")
bash "$MY_ROOT/run_verified_official.sh" \
  --output-folder "$EVALUATION_ROOT" \
  --descriptions-file "$DESCRIPTIONS" \
  "${RUN_FOLDERS[@]}"

RESULTS="$EVALUATION_ROOT/physics-IQ-benchmark-verified/results"
CSVS=()
for folder in "${RUN_FOLDERS[@]}"; do
  name="$(basename "${folder%/}")"
  csv="$RESULTS/$name.csv"
  [[ -s "$csv" ]] || { echo "Missing official result CSV: $csv" >&2; exit 1; }
  CSVS+=("$csv")
done

bash "$MY_ROOT/aggregate_verified_official.sh" \
  "${CSVS[@]}" \
  --save-csv "$EVALUATION_ROOT/verified_summary.csv"

printf 'Verified summary: %s\n' "$EVALUATION_ROOT/verified_summary.csv"
