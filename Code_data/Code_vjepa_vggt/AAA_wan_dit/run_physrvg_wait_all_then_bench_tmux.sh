#!/usr/bin/env bash
set -euo pipefail

# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physrvg_wait_all_then_bench_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/PhyRVG}"
RESULT_ROOTS_FILE="${RESULT_ROOTS_FILE:-${RESULT_ROOT}/rvg_leaf_folders.txt}"
INPUT_ALLOWLIST="${INPUT_ALLOWLIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
WAIT_SESSION="${WAIT_SESSION:-physrvg_wait_all_then_bench_20260724}"
BENCH_SESSION="${BENCH_SESSION:-bench_physrvg_after_complete_gpu0123456_20260724}"
BENCH_RUN_ROOT="${BENCH_RUN_ROOT:-${RESULT_ROOT}/_bench_runs/${BENCH_SESSION}}"
EXPECTED_ROOTS="${EXPECTED_ROOTS:-31}"
EXPECTED_CASES="${EXPECTED_CASES:-67}"
POLL_SECONDS="${POLL_SECONDS:-60}"

BENCH_LAUNCHER="${SCRIPT_DIR}/run_bench_physrvg_gpu0123456_tmux.sh"
GALLERY="${SCRIPT_DIR}/build_v2v_wan_case_gallery.py"
PLOT="${SCRIPT_DIR}/run_plot_dit_ablation_metrics.sh"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"

if [[ "${PHYSRVG_WAIT_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${WAIT_SESSION}" >&2
    exit 1
  fi
  tmux new-session -d -s "${WAIT_SESSION}" -n wait \
    "PHYSRVG_WAIT_INSIDE_TMUX=1 WAIT_SESSION='${WAIT_SESSION}' BENCH_SESSION='${BENCH_SESSION}' BENCH_RUN_ROOT='${BENCH_RUN_ROOT}' bash '$0'"
  echo "tmux wait session: ${WAIT_SESSION}"
  echo "benchmark session after gate: ${BENCH_SESSION}"
  echo "benchmark run root: ${BENCH_RUN_ROOT}"
  exit 0
fi

if [[ ! -s "${RESULT_ROOTS_FILE}" ]]; then
  echo "Missing or empty result-root list: ${RESULT_ROOTS_FILE}" >&2
  exit 2
fi
if [[ ! -s "${INPUT_ALLOWLIST}" ]]; then
  echo "Missing or empty input allowlist: ${INPUT_ALLOWLIST}" >&2
  exit 2
fi

mapfile -t RESULT_ROOTS < <(
  sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${RESULT_ROOTS_FILE}"
)
if [[ "${#RESULT_ROOTS[@]}" -ne "${EXPECTED_ROOTS}" ]]; then
  echo "Expected ${EXPECTED_ROOTS} result roots, got ${#RESULT_ROOTS[@]}" >&2
  exit 2
fi

echo "[wait] strict gate: ${EXPECTED_ROOTS} roots x ${EXPECTED_CASES} cases"
while true; do
  complete_roots=0
  total_mp4=0
  total_json=0
  for result_dir in "${RESULT_ROOTS[@]}"; do
    mp4_count=0
    json_count=0
    if [[ -d "${result_dir}" ]]; then
      mp4_count="$(find "${result_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
      json_count="$(
        find "${result_dir}" -maxdepth 1 -type f -name '*.json' \
          ! -name 'eval_summary_*.json' \
          ! -name 'summary.json' \
          ! -name 'result.json' \
          ! -name 'batch_manifest.json' | wc -l
      )"
    fi
    total_mp4=$((total_mp4 + mp4_count))
    total_json=$((total_json + json_count))
    if [[ "${mp4_count}" -ge "${EXPECTED_CASES}" && "${json_count}" -ge "${EXPECTED_CASES}" ]]; then
      complete_roots=$((complete_roots + 1))
    fi
  done

  printf '[wait] %s complete_roots=%s/%s mp4=%s/%s json=%s/%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${complete_roots}" "${EXPECTED_ROOTS}" \
    "${total_mp4}" "$((EXPECTED_ROOTS * EXPECTED_CASES))" \
    "${total_json}" "$((EXPECTED_ROOTS * EXPECTED_CASES))"
  if [[ "${complete_roots}" -eq "${EXPECTED_ROOTS}" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "[wait] all generation outputs are complete; refreshing gallery"
"${PYTHON_BIN}" "${GALLERY}" \
  --result-root "${RESULT_ROOT}" \
  --input-txt "${RESULT_ROOTS_FILE}" \
  --output-dir "${RESULT_ROOT}/_gallery"

echo "[wait] launching multi-GPU benchmark"
SESSION="${BENCH_SESSION}" \
RUN_ROOT="${BENCH_RUN_ROOT}" \
BASELINE_LIST="${RESULT_ROOTS_FILE}" \
INPUT_ALLOWLIST="${INPUT_ALLOWLIST}" \
bash "${BENCH_LAUNCHER}"

expected_workers=$((7 * (8 + 3)))
while true; do
  workers_done="$(find "${BENCH_RUN_ROOT}/state" -maxdepth 1 -type f -name '*.complete' | wc -l)"
  completed_tasks="$(wc -l < "${BENCH_RUN_ROOT}/completed_tasks.tsv")"
  failed_tasks="$(wc -l < "${BENCH_RUN_ROOT}/failed_tasks.tsv")"
  printf '[bench] %s workers=%s/%s completed_tasks=%s failed_tasks=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${workers_done}" "${expected_workers}" "${completed_tasks}" "${failed_tasks}"
  if [[ "${workers_done}" -eq "${expected_workers}" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "[bench] workers finished; refreshing plots and gallery"
INPUT_JSON_ALLOWLIST="${INPUT_ALLOWLIST}" \
EXPECTED_CASES="${EXPECTED_CASES}" \
bash "${PLOT}" "${RESULT_ROOTS_FILE}" "${RESULT_ROOT}/_metric_plots/rvg_leaf_folders"
"${PYTHON_BIN}" "${GALLERY}" \
  --result-root "${RESULT_ROOT}" \
  --input-txt "${RESULT_ROOTS_FILE}" \
  --output-dir "${RESULT_ROOT}/_gallery"

echo "[done] PhysRVG generation gate, metrics, plots, and gallery are complete"
exec bash
