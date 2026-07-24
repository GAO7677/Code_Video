#!/usr/bin/env bash
set -euo pipefail

# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_recompute_videophy2_generated_only_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_recompute_videophy2_generated_only_worker.sh"
VERIFY="${SCRIPT_DIR}/verify_videophy2_generated_only.py"
PLOT="${SCRIPT_DIR}/run_plot_dit_ablation_metrics.sh"
GALLERY="${SCRIPT_DIR}/build_v2v_wan_case_gallery.py"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py

PRIMARY_LIST="${PRIMARY_LIST:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/leaf_folders.txt}"
EXTRA_LIST="${EXTRA_LIST:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/PhyRVG/rvg_leaf_folders.txt}"
INPUT_ALLOWLIST="${INPUT_ALLOWLIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
RESULT_BASE="${RESULT_BASE:-/data/gaoya/AAA_test_video/0623/test/v2v_wan}"
EXPECTED_PRIMARY="${EXPECTED_PRIMARY:-32}"
EXPECTED_EXTRA="${EXPECTED_EXTRA:-31}"
EXPECTED_CASES="${EXPECTED_CASES:-67}"
GPUS=(0 1 2 3 4 5 6)
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-22000}"
SESSION="${SESSION:-recompute_videophy2_generated_only_20260724}"
RUN_ROOT="${RUN_ROOT:-${RESULT_BASE}/_bench_runs/${SESSION}}"
PRIOR_STATE_DIR="${PRIOR_STATE_DIR:-${RESULT_BASE}/PhyRVG/_bench_runs/bench_physrvg_31folders_gpu0123456_20260724/state}"
PRIOR_GPU_WORKERS="${PRIOR_GPU_WORKERS:-21}"
START_GATE="${RUN_ROOT}/start.ready"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ ! -s "${PRIMARY_LIST}" || ! -s "${EXTRA_LIST}" || ! -s "${INPUT_ALLOWLIST}" ]]; then
  echo "Missing result-root list or input allowlist" >&2
  exit 2
fi

mapfile -t PRIMARY_ROOTS < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${PRIMARY_LIST}")
mapfile -t EXTRA_ROOTS < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${EXTRA_LIST}")
if [[ "${#PRIMARY_ROOTS[@]}" -ne "${EXPECTED_PRIMARY}" || "${#EXTRA_ROOTS[@]}" -ne "${EXPECTED_EXTRA}" ]]; then
  echo "Unexpected root count: primary=${#PRIMARY_ROOTS[@]} extra=${#EXTRA_ROOTS[@]}" >&2
  exit 2
fi
ALL_ROOTS=("${PRIMARY_ROOTS[@]}" "${EXTRA_ROOTS[@]}")
if [[ "$(printf '%s\n' "${ALL_ROOTS[@]}" | sort -u | wc -l)" -ne "$((EXPECTED_PRIMARY + EXPECTED_EXTRA))" ]]; then
  echo "Duplicate result roots" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" "${RUN_ROOT}/state" "${RUN_ROOT}/task_summaries"
: > "${RUN_ROOT}/queues/videophy2.tsv"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"
printf '1\n' > "${RUN_ROOT}/queues/videophy2.cursor"
rm -f "${START_GATE}"
printf '%s\n' "${ALL_ROOTS[@]}" > "${RUN_ROOT}/all_leaf_folders.snapshot.txt"
cp "${PRIMARY_LIST}" "${RUN_ROOT}/primary_leaf_folders.snapshot.txt"
cp "${EXTRA_LIST}" "${RUN_ROOT}/extra_leaf_folders.snapshot.txt"
cp "${INPUT_ALLOWLIST}" "${RUN_ROOT}/input_allowlist.snapshot.txt"

"${PYTHON_BIN}" "${SUMMARY}" \
  --input-txt "${RUN_ROOT}/all_leaf_folders.snapshot.txt" \
  --output-csv "${RUN_ROOT}/metric_summary_before_recompute.csv" \
  --input-json-allowlist "${INPUT_ALLOWLIST}"

task_index=0
for root in "${PRIMARY_ROOTS[@]}"; do
  printf 'videophy2-%04d\t%s\t0\n' "${task_index}" "${root}" >> "${RUN_ROOT}/queues/videophy2.tsv"
  task_index=$((task_index + 1))
done
for root in "${EXTRA_ROOTS[@]}"; do
  printf 'videophy2-%04d\t%s\t1\n' "${task_index}" "${root}" >> "${RUN_ROOT}/queues/videophy2.tsv"
  task_index=$((task_index + 1))
done
TOTAL_TASKS="${task_index}"
TOTAL_WORKERS=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))

tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do prior_done=\$(find '${PRIOR_STATE_DIR}' -maxdepth 1 -type f -name 'g*_gpu*.complete' 2>/dev/null | wc -l); printf '[coordinator] waiting for prior PhyRVG GPU workers: %s/${PRIOR_GPU_WORKERS}\\n' \"\$prior_done\"; [ \"\$prior_done\" -ge '${PRIOR_GPU_WORKERS}' ] && break; sleep 60; done; touch '${START_GATE}'; while true; do workers=\$(find '${RUN_ROOT}/state' -maxdepth 1 -type f -name '*.complete' | wc -l); cursor=\$(cat '${RUN_ROOT}/queues/videophy2.cursor'); printf '[coordinator] workers=%s/${TOTAL_WORKERS} claimed=%s/${TOTAL_TASKS} completed=%s failed=%s\\n' \"\$workers\" \"\$((cursor - 1))\" \"\$(wc -l < '${RUN_ROOT}/completed_tasks.tsv')\" \"\$(wc -l < '${RUN_ROOT}/failed_tasks.tsv')\"; [ \"\$workers\" -eq '${TOTAL_WORKERS}' ] && break; sleep 30; done; '${PYTHON_BIN}' '${VERIFY}' --result-roots '${RUN_ROOT}/all_leaf_folders.snapshot.txt' --input-json-allowlist '${INPUT_ALLOWLIST}' --expected-cases '${EXPECTED_CASES}' --expected-context-frames 8 --output '${RUN_ROOT}/verification.json'; '${PYTHON_BIN}' '${SUMMARY}' --input-txt '${RUN_ROOT}/all_leaf_folders.snapshot.txt' --output-csv '${RUN_ROOT}/metric_summary_after_recompute.csv' --input-json-allowlist '${INPUT_ALLOWLIST}'; INPUT_JSON_ALLOWLIST='${INPUT_ALLOWLIST}' EXPECTED_CASES='${EXPECTED_CASES}' bash '${PLOT}' '${PRIMARY_LIST}' '${RESULT_BASE}/_metric_plots'; INPUT_JSON_ALLOWLIST='${INPUT_ALLOWLIST}' EXPECTED_CASES='${EXPECTED_CASES}' bash '${PLOT}' '${EXTRA_LIST}' '${RESULT_BASE}/PhyRVG/_metric_plots'; '${PYTHON_BIN}' '${GALLERY}' --result-root '${RESULT_BASE}' --output-dir '${RESULT_BASE}/_gallery'; printf '[coordinator] VideoPhy2 generated-only recompute complete\\n'; exec bash"

for gpu in "${GPUS[@]}"; do
  for worker_index in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    name="g${gpu}_vp${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' '${gpu}' '${name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}' '${EXPECTED_CASES}' '${GPU_MAX_USED_MIB}' '${START_GATE}' '${PRIOR_STATE_DIR}' '${PRIOR_GPU_WORKERS}'"
  done
done

tmux select-window -t "${SESSION}:coordinator"
echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
echo "tasks: ${TOTAL_TASKS}; workers: ${TOTAL_WORKERS} (${WORKERS_PER_GPU} per GPU)"
echo "GPU start threshold: memory.used <= ${GPU_MAX_USED_MIB} MiB"
echo "PhyRVG overwrite gate: ${PRIOR_GPU_WORKERS} prior GPU workers complete"
