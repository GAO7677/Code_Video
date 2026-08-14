#!/usr/bin/env bash
# Resume the formal run from its last complete optimizer/RNG/LoRA snapshot.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
CONFIG="${PROJECT_DIR}/configs/resume_full_sa_object_slot_dedup_xssc50k_xssc_loss_step500_gpu56.json"
RUN_TAG="${RUN_TAG:-resume_step000500_gpu56_20260814T095638Z}"

exec "${PYTHON_BIN}" "${EXPERIMENT_ROOT}/launch_from_config.py" \
  "${CONFIG}" --run-tag "${RUN_TAG}"
