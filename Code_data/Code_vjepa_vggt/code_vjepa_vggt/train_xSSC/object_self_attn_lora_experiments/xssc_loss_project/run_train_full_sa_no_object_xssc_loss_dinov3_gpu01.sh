#!/usr/bin/env bash
# Foreground formal launcher. The stable run tag matches the watcher root.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
CONFIG="${PROJECT_DIR}/configs/full_sa_no_object_xssc_loss_dinov3_movic_step50000.json"
RUN_TAG="${RUN_TAG:-formal_gpu01}"

exec "${PYTHON_BIN}" "${EXPERIMENT_ROOT}/launch_from_config.py" \
  "${CONFIG}" --run-tag "${RUN_TAG}"
