#!/usr/bin/env bash
set -euo pipefail

REPO="/home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main"

if (($# == 0)); then
  cat >&2 <<'EOF'
Usage:
  aggregate_verified_official.sh RUN1.csv [RUN2.csv RUN3.csv RUN4.csv] [official save options]

Examples of official save options:
  --save-csv /path/scores.csv --model-name MODEL
  --save-latex /path/table.tex
EOF
  exit 2
fi

for arg in "$@"; do
  [[ "$arg" != "--score-type" && "$arg" != "--score-type="* ]] || {
    printf 'Error: score type is fixed to verified by this wrapper.\n' >&2
    exit 2
  }
done

command -v uv >/dev/null 2>&1 || {
  printf 'Error: uv is not installed or not on PATH.\n' >&2
  exit 1
}

CMD=(uv run physiq/aggregate_runs_from_csvs.py "$@" --score-type verified)
printf 'Official command:'
printf ' %q' "${CMD[@]}"
printf '\n'

cd "$REPO"
exec "${CMD[@]}"
