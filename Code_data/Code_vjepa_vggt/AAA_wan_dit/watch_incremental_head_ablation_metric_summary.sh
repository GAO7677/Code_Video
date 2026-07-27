#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 RUN_ROOT INPUT_ALLOWLIST EXPECTED_WORKERS" >&2
  exit 2
fi

RUN_ROOT="$1"
INPUT_ALLOWLIST="$2"
EXPECTED_WORKERS="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
VIS_ROOT=/data/gaoya/agent-data/outputs/wan_attention_experiment_visualizations/head_ablation_test5
CONFIG="${SCRIPT_DIR}/head_ablation_allblocks_test5_gpu56.json"

while true; do
  completed="$(wc -l < "${RUN_ROOT}/completed_tasks.tsv")"
  if [[ "${completed}" -gt 0 && -s "${RUN_ROOT}/leaf_folders_incremental.txt" ]]; then
    "${PYTHON}" "${SUMMARY}" \
      --input-txt "${RUN_ROOT}/leaf_folders_incremental.txt" \
      --output-csv "${RUN_ROOT}/metric_summary_partial.csv" \
      --input-json-allowlist "${INPUT_ALLOWLIST}"
    "${PYTHON}" "${SCRIPT_DIR}/visualize_head_ablation_metric_curves.py" \
      --config "${CONFIG}" \
      --metric-summary "${RUN_ROOT}/metric_summary_partial.csv" \
      --output-dir "${VIS_ROOT}"
  fi
  workers="$(find "${RUN_ROOT}/state" -name '*.complete' -type f | wc -l)"
  echo "[incremental-summary] completed_tasks=${completed} workers=${workers}/${EXPECTED_WORKERS}"
  if [[ -f "${RUN_ROOT}/enqueue.complete" && "${workers}" -eq "${EXPECTED_WORKERS}" ]]; then
    break
  fi
  sleep 300
done
