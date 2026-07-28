#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/watch_test5_st_phased_gallery.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${CONFIG:-${SCRIPT_DIR}/test5_st_phased_seed851.json}"
INTERVAL="${INTERVAL:-15}"

while true; do
  "${PYTHON}" "${SCRIPT_DIR}/build_test5_st_phased_gallery.py" \
    --config "${CONFIG}"
  sleep "${INTERVAL}"
done
