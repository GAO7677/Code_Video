#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
REPO=/home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main
export PATH="/home/gaoya/miniconda3/envs/wan-cu128/bin:$PATH"
export UV_CACHE_DIR=/data/gaoya/agent-data/cache/uv
export UV_PROJECT_ENVIRONMENT=/data/gaoya/agent-data/cache/envs/physics-iq-verified
mkdir -p "$UV_CACHE_DIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if ! command -v uv >/dev/null 2>&1; then
  "$PYTHON" -m pip install --upgrade uv
fi

cd "$REPO"
"$(command -v uv)" sync

printf 'Official benchmark environment: %s\n' "$UV_PROJECT_ENVIRONMENT"
