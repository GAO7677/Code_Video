#!/usr/bin/env bash
# Usage:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_train_from_config.sh \
#   /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/configs/object_only.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 CONFIG.json [--run-tag TAG] [--dry-run|--validate-only]" >&2
  exit 2
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/launch_from_config.py" "$@"
