#!/usr/bin/env bash
set -euo pipefail

# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_recompute_pmf_aligned_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_recompute_pmf_aligned_worker.sh"
VERIFY="${SCRIPT_DIR}/verify_pmf_time_alignment.py"
PLOT="${SCRIPT_DIR}/run_plot_dit_ablation_metrics.sh"
GALLERY="${SCRIPT_DIR}/build_v2v_wan_case_gallery.py"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py

RESULT_ROOTS_FILE="${RESULT_ROOTS_FILE:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/leaf_folders.txt}"
EXTRA_RESULT_ROOTS_FILE="${EXTRA_RESULT_ROOTS_FILE:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/PhyRVG/rvg_leaf_folders.txt}"
INPUT_ALLOWLIST="${INPUT_ALLOWLIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
RESULT_BASE="${RESULT_BASE:-/data/gaoya/AAA_test_video/0623/test/v2v_wan}"
EXPECTED_ROOTS="${EXPECTED_ROOTS:-32}"
EXPECTED_EXTRA_ROOTS="${EXPECTED_EXTRA_ROOTS:-31}"
EXPECTED_CASES="${EXPECTED_CASES:-67}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SESSION="${SESSION:-recompute_pmf_timestamp_aligned_20260724}"
RUN_ROOT="${RUN_ROOT:-${RESULT_BASE}/_bench_runs/${SESSION}}"
MAX_ACTIVE_PRIOR_METRIC_PROCESSES="${MAX_ACTIVE_PRIOR_METRIC_PROCESSES:-24}"
PRIOR_STATE_DIR="${PRIOR_STATE_DIR:-${RESULT_BASE}/PhyRVG/_bench_runs/bench_physrvg_31folders_gpu0123456_20260724/state}"
PRIOR_CPU_WORKERS="${PRIOR_CPU_WORKERS:-56}"
START_GATE="${RUN_ROOT}/start.ready"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ ! -s "${RESULT_ROOTS_FILE}" || ! -s "${EXTRA_RESULT_ROOTS_FILE}" || ! -s "${INPUT_ALLOWLIST}" ]]; then
  echo "Missing result-root list, extra result-root list, or input allowlist" >&2
  exit 2
fi

mapfile -t RESULT_ROOTS < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${RESULT_ROOTS_FILE}")
mapfile -t EXTRA_RESULT_ROOTS < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${EXTRA_RESULT_ROOTS_FILE}")
if [[ "${#RESULT_ROOTS[@]}" -ne "${EXPECTED_ROOTS}" ]]; then
  echo "Expected ${EXPECTED_ROOTS} result roots, got ${#RESULT_ROOTS[@]}" >&2
  exit 2
fi
if [[ "${#EXTRA_RESULT_ROOTS[@]}" -ne "${EXPECTED_EXTRA_ROOTS}" ]]; then
  echo "Expected ${EXPECTED_EXTRA_ROOTS} extra result roots, got ${#EXTRA_RESULT_ROOTS[@]}" >&2
  exit 2
fi
ALL_RESULT_ROOTS=("${RESULT_ROOTS[@]}" "${EXTRA_RESULT_ROOTS[@]}")
EXPECTED_ALL_ROOTS=$((EXPECTED_ROOTS + EXPECTED_EXTRA_ROOTS))
if [[ "$(printf '%s\n' "${ALL_RESULT_ROOTS[@]}" | sort -u | wc -l)" -ne "${EXPECTED_ALL_ROOTS}" ]]; then
  echo "Result-root list contains duplicate paths" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" "${RUN_ROOT}/state" "${RUN_ROOT}/task_summaries"
: > "${RUN_ROOT}/queues/pmf.tsv"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"
printf '1\n' > "${RUN_ROOT}/queues/pmf.cursor"
cp "${RESULT_ROOTS_FILE}" "${RUN_ROOT}/leaf_folders.snapshot.txt"
cp "${EXTRA_RESULT_ROOTS_FILE}" "${RUN_ROOT}/extra_leaf_folders.snapshot.txt"
printf '%s\n' "${ALL_RESULT_ROOTS[@]}" > "${RUN_ROOT}/all_leaf_folders.snapshot.txt"
cp "${INPUT_ALLOWLIST}" "${RUN_ROOT}/input_allowlist.snapshot.txt"

"${PYTHON_BIN}" "${SUMMARY}" \
  --input-txt "${RUN_ROOT}/all_leaf_folders.snapshot.txt" \
  --output-csv "${RUN_ROOT}/metric_summary_before_recompute.csv" \
  --input-json-allowlist "${INPUT_ALLOWLIST}"

task_index=0
for metric in pmf_with_context pmf_without_context; do
  for root in "${RESULT_ROOTS[@]}"; do
    printf 'pmf-%04d\t%s\t%s\t0\n' "${task_index}" "${metric}" "${root}" >> "${RUN_ROOT}/queues/pmf.tsv"
    task_index=$((task_index + 1))
  done
done
for metric in pmf_with_context pmf_without_context; do
  for root in "${EXTRA_RESULT_ROOTS[@]}"; do
    printf 'pmf-%04d\t%s\t%s\t1\n' "${task_index}" "${metric}" "${root}" >> "${RUN_ROOT}/queues/pmf.tsv"
    task_index=$((task_index + 1))
  done
done
TOTAL_TASKS="${task_index}"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do active=\$(pgrep -fc '[A]AAinfer/bench.py --metric (physics_iq_with_context|physics_iq_without_context|pmf_with_context|pmf_without_context)' || true); printf '[coordinator] waiting for active prior metric processes <= ${MAX_ACTIVE_PRIOR_METRIC_PROCESSES}: %s\\n' \"\$active\"; [ \"\$active\" -le '${MAX_ACTIVE_PRIOR_METRIC_PROCESSES}' ] && break; sleep 60; done; touch '${START_GATE}'; while true; do workers=\$(find '${RUN_ROOT}/state' -maxdepth 1 -type f -name '*.complete' | wc -l); cursor=\$(cat '${RUN_ROOT}/queues/pmf.cursor'); printf '[coordinator] workers=%s/${NUM_WORKERS} claimed=%s/${TOTAL_TASKS} completed=%s failed=%s\\n' \"\$workers\" \"\$((cursor - 1))\" \"\$(wc -l < '${RUN_ROOT}/completed_tasks.tsv')\" \"\$(wc -l < '${RUN_ROOT}/failed_tasks.tsv')\"; [ \"\$workers\" -eq '${NUM_WORKERS}' ] && break; sleep 30; done; '${PYTHON_BIN}' '${VERIFY}' --result-roots '${RUN_ROOT}/all_leaf_folders.snapshot.txt' --input-json-allowlist '${INPUT_ALLOWLIST}' --expected-cases '${EXPECTED_CASES}' --output '${RUN_ROOT}/verification.json'; '${PYTHON_BIN}' '${SUMMARY}' --input-txt '${RUN_ROOT}/all_leaf_folders.snapshot.txt' --output-csv '${RUN_ROOT}/metric_summary_after_recompute.csv' --input-json-allowlist '${INPUT_ALLOWLIST}'; INPUT_JSON_ALLOWLIST='${INPUT_ALLOWLIST}' EXPECTED_CASES='${EXPECTED_CASES}' bash '${PLOT}' '${RESULT_ROOTS_FILE}' '${RESULT_BASE}/_metric_plots'; INPUT_JSON_ALLOWLIST='${INPUT_ALLOWLIST}' EXPECTED_CASES='${EXPECTED_CASES}' bash '${PLOT}' '${EXTRA_RESULT_ROOTS_FILE}' '${RESULT_BASE}/PhyRVG/_metric_plots'; '${PYTHON_BIN}' '${GALLERY}' --result-root '${RESULT_BASE}' --output-dir '${RESULT_BASE}/_gallery'; printf '[coordinator] recompute and refresh complete\\n'; exec bash"

for worker_index in $(seq 0 $((NUM_WORKERS - 1))); do
  name="pmf${worker_index}"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${WORKER}' '${name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}' '${START_GATE}' '${PRIOR_STATE_DIR}' '${PRIOR_CPU_WORKERS}'"
done

tmux select-window -t "${SESSION}:coordinator"
echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
echo "result roots: ${EXPECTED_ALL_ROOTS} (${EXPECTED_ROOTS} primary + ${EXPECTED_EXTRA_ROOTS} PhyRVG)"
echo "tasks: ${TOTAL_TASKS}; workers: ${NUM_WORKERS}"
echo "start threshold: active prior Physics-IQ/PMF processes <= ${MAX_ACTIVE_PRIOR_METRIC_PROCESSES}"
echo "PhyRVG overwrite gate: ${PRIOR_CPU_WORKERS} prior CPU workers complete"
