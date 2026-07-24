#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_plot_dit_ablation_metrics.sh

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_dit_ablation_metrics.py" \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v_wan \
  --input-json-allowlist /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
  --expected-cases 67 \
  --output-dir /data/gaoya/AAA_test_video/0623/test/v2v_wan/_metric_plots
