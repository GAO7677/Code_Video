#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 RUN_ROOT OUTPUT_BASE INPUT_LIST NUM_GEN NUM_CPU NUM_GPU NUM_COSMOS" >&2
  exit 2
fi

RUN_ROOT="$1"
OUTPUT_BASE="$2"
INPUT_LIST="$3"
NUM_GEN="$4"
NUM_CPU="$5"
NUM_GPU="$6"
NUM_COSMOS="$7"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
MANAGER="${SCRIPT_DIR}/manage_remaining_block_pipeline.py"
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
PLOT="${SCRIPT_DIR}/run_plot_dit_ablation_metrics.sh"
METRICS="${RUN_ROOT}/metrics"

trap 'touch "${RUN_ROOT}/pipeline.failed"' ERR

wait_count() {
  local pattern="$1" expected="$2" label="$3" directory="$4" count
  while true; do
    count="$(find "${directory}" -maxdepth 1 -type f -name "${pattern}" | wc -l)"
    printf '[coordinator] %s=%s/%s\n' "${label}" "${count}" "${expected}"
    [[ "${count}" -eq "${expected}" ]] && return 0
    sleep 60
  done
}

wait_count '*.complete' "${NUM_GEN}" "generation_workers" "${RUN_ROOT}/generation/state"
if [[ -s "${RUN_ROOT}/generation/failed.tsv" ]]; then
  touch "${RUN_ROOT}/pipeline.failed"
  echo "[coordinator] generation failures detected" >&2
  exit 1
fi
if [[ "$(wc -l < "${RUN_ROOT}/generation/completed.tsv")" -ne 240 ]]; then
  touch "${RUN_ROOT}/pipeline.failed"
  echo "[coordinator] expected 240 completed generation jobs" >&2
  exit 1
fi

"${PYTHON_BIN}" "${MANAGER}" prepare-metrics \
  --output-base "${OUTPUT_BASE}" \
  --input-list "${INPUT_LIST}" \
  --all-roots "${RUN_ROOT}/all_leaf_folders.txt" \
  --new-roots "${RUN_ROOT}/new_leaf_folders.txt" \
  --queue-dir "${METRICS}/queues" \
  --report "${RUN_ROOT}/metric_manifest.json"

for kind in cpu gpu_common videophy2 cosmos retry; do
  printf '1\n' > "${METRICS}/queues/${kind}.cursor"
  : > "${METRICS}/queues/${kind}.lock"
done
: > "${METRICS}/completed_tasks.tsv"
: > "${METRICS}/failed_tasks.tsv"

touch "${RUN_ROOT}/cpu.ready"
wait_count 'cpu_*.stage_complete' "${NUM_CPU}" "cpu_workers" "${METRICS}/state"

touch "${RUN_ROOT}/gpu_common.ready"
wait_count 'gpu_*.stage_complete' "${NUM_GPU}" "gpu_common_workers" "${METRICS}/state"

touch "${RUN_ROOT}/videophy2.ready"
wait_count 'vp_*.stage_complete' "${NUM_GPU}" "videophy2_workers" "${METRICS}/state"

touch "${RUN_ROOT}/cosmos.ready"
wait_count 'cosmos_*.stage_complete' "${NUM_COSMOS}" "cosmos_workers" "${METRICS}/state"

"${PYTHON_BIN}" "${MANAGER}" build-retry \
  --all-roots "${RUN_ROOT}/all_leaf_folders.txt" \
  --input-list "${INPUT_LIST}" \
  --queue "${METRICS}/queues/retry.tsv" \
  --report "${RUN_ROOT}/retry_manifest.json"

if [[ -s "${METRICS}/queues/retry.tsv" ]]; then
  touch "${RUN_ROOT}/retry.ready"
  wait_count 'retry_*.stage_complete' "${NUM_GPU}" "retry_workers" "${METRICS}/state"
else
  touch "${RUN_ROOT}/retry.not_needed"
  touch "${RUN_ROOT}/retry.ready"
fi

"${PYTHON_BIN}" "${MANAGER}" verify-all \
  --all-roots "${RUN_ROOT}/all_leaf_folders.txt" \
  --input-list "${INPUT_LIST}" \
  --output "${RUN_ROOT}/verification_final.json"

"${PYTHON_BIN}" "${SUMMARY}" \
  --input-txt "${RUN_ROOT}/all_leaf_folders.txt" \
  --output-csv "${RUN_ROOT}/metric_summary.csv" \
  --input-json-allowlist "${INPUT_LIST}"

INPUT_JSON_ALLOWLIST="${INPUT_LIST}" EXPECTED_CASES=67 \
  bash "${PLOT}" "${RUN_ROOT}/all_leaf_folders.txt" \
  "${OUTPUT_BASE}/_metric_plots/all_blocks"

touch "${RUN_ROOT}/pipeline.complete"
echo "[coordinator] all remaining blocks, metrics, verification, and plots complete"
