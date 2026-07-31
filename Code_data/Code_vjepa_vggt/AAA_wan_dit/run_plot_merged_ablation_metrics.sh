#!/usr/bin/env bash
# Merge the two ablation metric-stats CSVs and plot per-method curves.
#
# Run (defaults to the two known runs):
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_plot_merged_ablation_metrics.sh
#
# Override CSVs / output dir explicitly:
# bash .../run_plot_merged_ablation_metrics.sh \
#   --csv /path/a/dit_ablation_metric_stats.csv \
#   --csv /path/b/dit_ablation_metric_stats.csv \
#   --output-dir /data/gaoya/.../merged

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_merged_ablation_metrics.py" "$@"
