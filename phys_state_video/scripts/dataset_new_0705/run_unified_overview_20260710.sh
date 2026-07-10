#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/gaoya/Code_Video/phys_state_video}"
RIGID_ROOT="${RIGID_ROOT:-/data/gaoya/agent-data/outputs/dataset_new_0705/AAA_check_0710}"
MPM_ROOT="${MPM_ROOT:-/data/gaoya/agent-data/outputs/dataset_new_0705/mpm_preview_batch_20260710}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/dataset_new_0705/unified_overview_20260710}"
PORT="${PORT:-18830}"

PYTHONPATH="${PROJECT_ROOT}/scripts:/home/gaoya/Code_Video" \
"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/dataset_new_0705/build_unified_overview_page.py" \
  --rigid-root "${RIGID_ROOT}" \
  --mpm-root "${MPM_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --port "${PORT}"

cd "${OUTPUT_ROOT}"
exec "${PYTHON_BIN}" -m http.server "${PORT}" --bind 127.0.0.1
