#!/usr/bin/env bash
# Foreground launcher. The config inherits the Full-SA no-object xSSC-loss setup
# and replaces Full-SA with the fixed step-39 Physics-IQ67 PCK@32 Top100 heads.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
CONFIG="${PROJECT_DIR}/configs/t_head_pck32_s039_latest3350_top100_no_object_xssc_loss_dinov3_movic_step50000.json"
RUN_TAG="${RUN_TAG:-formal_gpu01}"

export PYTHONNOUSERSITE=1

exec "${PYTHON_BIN}" "${EXPERIMENT_ROOT}/launch_from_config.py" \
  "${CONFIG}" --run-tag "${RUN_TAG}"
