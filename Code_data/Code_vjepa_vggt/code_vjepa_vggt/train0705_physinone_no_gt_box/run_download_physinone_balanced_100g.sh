#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"
CACHE_ROOT="${CACHE_ROOT:-/data/gaoya/agent-data/cache/huggingface}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/dataset/vLAR-PhysInOne/TrainBalanced100G}"
ASSETS_ROOT="${ASSETS_ROOT:-/data/gaoya/dataset/vLAR-PhysInOne/vLAR-PhysInOne-assets/assets}"

mkdir -p "$CACHE_ROOT"

CMD=(
  env
  HF_ENDPOINT="$HF_ENDPOINT_VALUE"
  HF_HOME="$CACHE_ROOT"
  HF_HUB_CACHE="$CACHE_ROOT/hub"
  "$PYTHON_BIN"
  "$SCRIPT_DIR/download_physinone_balanced_subset.py"
  --assets-root "$ASSETS_ROOT"
  --output-root "$OUTPUT_ROOT"
  --cache-dir "$CACHE_ROOT/hub"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  CMD+=(--hf-token "$HF_TOKEN")
fi

CMD+=("$@")

printf 'Resolved command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
