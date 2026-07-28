#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${CONFIG:-${SCRIPT_DIR}/common22_public_head_ablation.json}"
INTERVAL="${INTERVAL:-10}"

while true; do
  PYTHONPATH="${SCRIPT_DIR}" "${PYTHON}" \
    "${SCRIPT_DIR}/build_common22_public_head_multiseed_gallery.py" \
    --config "${CONFIG}" || true
  sleep "${INTERVAL}"
done
