#!/usr/bin/env bash
# Usage:
# bash run_train_from_config_vjepa_loss.sh configs/formal_full_sa_no_object_gpu27_vjepa_loss.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 CONFIG.json [--run-tag TAG] [--dry-run|--validate-only]" >&2
  exit 2
fi

exec "${PYTHON_BIN}" "${EXPERIMENT_ROOT}/launch_from_config.py" "$@"
