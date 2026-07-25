#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "Usage: $0 RUN_ROOT PRIMARY_LIST XSSC_LIST PHYRVG_LIST INPUT_ALLOWLIST VIDEO_WORKERS COSMOS_WORKERS RESULT_BASE" >&2
  exit 2
fi

RUN_ROOT="$1"
PRIMARY_LIST="$2"
XSSC_LIST="$3"
PHYRVG_LIST="$4"
INPUT_ALLOWLIST="$5"
VIDEO_WORKERS="$6"
COSMOS_WORKERS="$7"
RESULT_BASE="$8"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
VERIFY_VIDEO="${SCRIPT_DIR}/verify_videophy2_generated_only.py"
VERIFY_METRIC="${SCRIPT_DIR}/verify_metric_completion.py"
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
PLOT="${SCRIPT_DIR}/run_plot_dit_ablation_metrics.sh"
GALLERY="${SCRIPT_DIR}/build_v2v_wan_case_gallery.py"

wait_for_workers() {
  local pattern="$1" expected="$2" label="$3" count
  while true; do
    count="$(find "${RUN_ROOT}/state" -maxdepth 1 -type f -name "${pattern}" | wc -l)"
    printf '[coordinator] %s workers: %s/%s\n' "${label}" "${count}" "${expected}"
    if [[ "${count}" -eq "${expected}" ]]; then
      return 0
    fi
    sleep 30
  done
}

copy_task_summaries() {
  local queue="$1" metric="$2" task_id result_root unused
  while IFS=$'\t' read -r task_id unused result_root; do
    if [[ "${metric}" == "videophy2" ]]; then
      result_root="${unused}"
    fi
    cp "${RUN_ROOT}/task_summaries/${task_id}.json" "${result_root}/eval_summary_${metric}.json"
  done < "${queue}"
}

wait_for_workers 'g*_vp*.complete' "${VIDEO_WORKERS}" "VideoPhy2"
if grep -q '^videophy2-' "${RUN_ROOT}/failed_tasks.tsv"; then
  echo "[coordinator] VideoPhy2 task failure detected" >&2
  exit 1
fi

"${PYTHON_BIN}" "${VERIFY_VIDEO}" \
  --result-roots "${PRIMARY_LIST}" \
  --input-json-allowlist "${INPUT_ALLOWLIST}" \
  --expected-cases 67 \
  --expected-context-frames 8 \
  --output "${RUN_ROOT}/verification_videophy2.json"
copy_task_summaries "${RUN_ROOT}/queues/videophy2.tsv" videophy2

touch "${RUN_ROOT}/cosmos.start.ready"
wait_for_workers 'g*_cosmos*.complete' "${COSMOS_WORKERS}" "Cosmos"
if grep -q '^cosmos-' "${RUN_ROOT}/failed_tasks.tsv"; then
  echo "[coordinator] Cosmos task failure detected" >&2
  exit 1
fi

"${PYTHON_BIN}" "${VERIFY_METRIC}" \
  --result-roots "${XSSC_LIST}" \
  --input-json-allowlist "${INPUT_ALLOWLIST}" \
  --metric cosmos_reason1 \
  --required-field score \
  --expected-cases 67 \
  --output "${RUN_ROOT}/verification_cosmos_reason1.json"
copy_task_summaries "${RUN_ROOT}/queues/cosmos.tsv" cosmos_reason1

"${PYTHON_BIN}" "${SUMMARY}" \
  --input-txt "${PRIMARY_LIST}" \
  --output-csv "${RUN_ROOT}/metric_summary_primary_after_recompute.csv" \
  --input-json-allowlist "${INPUT_ALLOWLIST}"
"${PYTHON_BIN}" "${SUMMARY}" \
  --input-txt "${PHYRVG_LIST}" \
  --output-csv "${RUN_ROOT}/metric_summary_physrvg_after_recompute.csv" \
  --input-json-allowlist "${INPUT_ALLOWLIST}"

INPUT_JSON_ALLOWLIST="${INPUT_ALLOWLIST}" EXPECTED_CASES=67 \
  bash "${PLOT}" "${PRIMARY_LIST}" "${RESULT_BASE}/_metric_plots"
INPUT_JSON_ALLOWLIST="${INPUT_ALLOWLIST}" EXPECTED_CASES=67 \
  bash "${PLOT}" "${PHYRVG_LIST}" "${RESULT_BASE}/PhyRVG/_metric_plots"
"${PYTHON_BIN}" "${GALLERY}" \
  --result-root "${RESULT_BASE}" \
  --output-dir "${RESULT_BASE}/_gallery"

touch "${RUN_ROOT}/pipeline.complete"
echo "[coordinator] VideoPhy2, Cosmos, summaries, plots, and gallery are complete"
