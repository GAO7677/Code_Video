#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "Usage: $0 RUN_ROOT NUM_VP_WORKERS NUM_COSMOS_WORKERS NUM_COMMON_WORKERS INPUT_LIST LEAF_LIST OUTPUT_BASE PIPELINE_ROOT" >&2
  exit 2
fi

RUN_ROOT="$1"
NUM_VP_WORKERS="$2"
NUM_COSMOS_WORKERS="$3"
NUM_COMMON_WORKERS="$4"
INPUT_LIST="$5"
LEAF_LIST="$6"
OUTPUT_BASE="$7"
PIPELINE_ROOT="$8"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
PLOT="${SCRIPT_DIR}/run_plot_dit_ablation_metrics.sh"
VERIFY="${SCRIPT_DIR}/verify_failed_metric_retry.py"
PLOT_DIR="${OUTPUT_BASE}/_metric_plots/leaf_folders"

wait_for_workers() {
  local pattern="$1" expected="$2" label="$3" count
  while true; do
    count="$(find "${RUN_ROOT}/state" -maxdepth 1 -type f -name "${pattern}" | wc -l)"
    printf '[retry-coordinator] %s workers=%s/%s completed_tasks=%s process_failures=%s\n' \
      "${label}" "${count}" "${expected}" \
      "$(wc -l < "${RUN_ROOT}/completed_tasks.tsv")" \
      "$(wc -l < "${RUN_ROOT}/failed_tasks.tsv")"
    [[ "${count}" -eq "${expected}" ]] && return 0
    sleep 30
  done
}

touch "${RUN_ROOT}/videophy2.ready"
wait_for_workers 'vp_g*.complete' "${NUM_VP_WORKERS}" "VideoPhy2"

touch "${RUN_ROOT}/cosmos.ready"
wait_for_workers 'cosmos_g*.complete' "${NUM_COSMOS_WORKERS}" "Cosmos"

touch "${RUN_ROOT}/gpu_common.ready"
wait_for_workers 'common_g*.complete' "${NUM_COMMON_WORKERS}" "GPU-common"

if [[ -s "${RUN_ROOT}/failed_tasks.tsv" ]]; then
  touch "${RUN_ROOT}/retry.failed"
  echo "[retry-coordinator] metric subprocess failures detected" >&2
  exit 1
fi

"${PYTHON_BIN}" "${VERIFY}" \
  --manifest "${RUN_ROOT}/retry_manifest.json" \
  --summary-dir "${RUN_ROOT}/task_summaries" \
  --expected-cases 5 \
  --output "${RUN_ROOT}/verification_retry_summaries.json"

"${PYTHON_BIN}" "${SUMMARY}" \
  --input-txt "${LEAF_LIST}" \
  --output-csv "${RUN_ROOT}/metric_summary_after_retry.csv" \
  --input-json-allowlist "${INPUT_LIST}"

INPUT_JSON_ALLOWLIST="${INPUT_LIST}" EXPECTED_CASES=5 \
  bash "${PLOT}" "${LEAF_LIST}" "${PLOT_DIR}"

"${PYTHON_BIN}" "${VERIFY}" \
  --manifest "${RUN_ROOT}/retry_manifest.json" \
  --summary-dir "${RUN_ROOT}/task_summaries" \
  --expected-cases 5 \
  --stats-csv "${PLOT_DIR}/dit_ablation_metric_stats.csv" \
  --output "${RUN_ROOT}/verification_final.json"

touch "${RUN_ROOT}/retry.complete"
touch "${PIPELINE_ROOT}/pipeline.complete"
echo "[retry-coordinator] all missing metrics recomputed and plots refreshed"
