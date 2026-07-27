#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 RUN_ROOT OUTPUT_BASE INPUT_LIST SESSION NUM_GEN_JOBS NUM_GEN_WORKERS" >&2
  exit 2
fi

RUN_ROOT="$1"
OUTPUT_BASE="$2"
INPUT_LIST="$3"
SESSION="$4"
NUM_GEN_JOBS="$5"
NUM_GEN_WORKERS="$6"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
MANAGER="${SCRIPT_DIR}/manage_remaining_block_pipeline.py"
PREPARE="${SCRIPT_DIR}/prepare_whole_block_pipeline.py"
METRIC_WORKER="${SCRIPT_DIR}/run_remaining_blocks_queue_worker.sh"
PLOT="${SCRIPT_DIR}/plot_dit_ablation_metrics.py"
GALLERY="${SCRIPT_DIR}/build_v2v_wan_case_gallery.py"
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
METRICS="${RUN_ROOT}/metrics"
CURRENT_PLOTS="${OUTPUT_BASE}/_metric_plots/current_complete_only"
WHOLE_PLOTS="${OUTPUT_BASE}/_metric_plots/whole_block_complete"

trap 'touch "${RUN_ROOT}/pipeline.failed"' ERR

wait_for_workers() {
  local pattern="$1" expected="$2" label="$3" directory="$4" count
  while true; do
    count="$(find "${directory}" -maxdepth 1 -type f -name "${pattern}" | wc -l)"
    printf '[whole-block] %s=%s/%s\n' "${label}" "${count}" "${expected}"
    [[ "${count}" -eq "${expected}" ]] && return 0
    sleep 30
  done
}

launch_metric_round() {
  local round="$1"
  local raw_queue="${METRICS}/queues/${round}_all.tsv"
  local report="${RUN_ROOT}/${round}_manifest.json"
  local cpu_kind="${round}_cpu"
  local gpu_kind="${round}_gpu"
  local cpu_queue="${METRICS}/queues/${cpu_kind}.tsv"
  local gpu_queue="${METRICS}/queues/${gpu_kind}.tsv"
  local cpu_workers=6
  local gpu_workers=3

  "${PYTHON_BIN}" "${MANAGER}" build-retry \
    --all-roots "${RUN_ROOT}/whole_block_leaf_folders.txt" \
    --input-list "${INPUT_LIST}" \
    --queue "${raw_queue}" \
    --report "${report}"

  awk -F $'\t' -v OFS=$'\t' -v prefix="${round}" \
    '$2 ~ /^(physics_iq_with_context|physics_iq_without_context|physics_iq_verified_proxy|pmf_with_context|pmf_without_context)$/ \
      {$1 = prefix "-" $1; print}' \
    "${raw_queue}" > "${cpu_queue}"
  awk -F $'\t' -v OFS=$'\t' -v prefix="${round}" \
    '$2 !~ /^(physics_iq_with_context|physics_iq_without_context|physics_iq_verified_proxy|pmf_with_context|pmf_without_context)$/ \
      {$1 = prefix "-" $1; print}' \
    "${raw_queue}" > "${gpu_queue}"

  for kind in "${cpu_kind}" "${gpu_kind}"; do
    printf '1\n' > "${METRICS}/queues/${kind}.cursor"
    : > "${METRICS}/queues/${kind}.lock"
  done

  printf '[whole-block] metric_round=%s cpu_tasks=%s gpu_tasks=%s\n' \
    "${round}" "$(wc -l < "${cpu_queue}")" "$(wc -l < "${gpu_queue}")"

  for gpu in 0 1 2; do
    for index in 0 1; do
      name="${round}_cpu_g${gpu}_${index}"
      tmux new-window -d -t "${SESSION}" -n "${name}" \
        "bash '${METRIC_WORKER}' '${gpu}' '${cpu_kind}' '${name}' '${METRICS}' '${INPUT_LIST}'; exec bash"
    done
    name="${round}_gpu_g${gpu}"
    tmux new-window -d -t "${SESSION}" -n "${name}" \
      "bash '${METRIC_WORKER}' '${gpu}' '${gpu_kind}' '${name}' '${METRICS}' '${INPUT_LIST}'; exec bash"
  done

  wait_for_workers "${round}_cpu_g*.complete" "${cpu_workers}" \
    "${round}_cpu_workers" "${METRICS}/state"
  wait_for_workers "${round}_gpu_g*.complete" "${gpu_workers}" \
    "${round}_gpu_workers" "${METRICS}/state"
}

wait_for_workers 'whole_gen_g*.complete' "${NUM_GEN_WORKERS}" \
  "generation_workers" "${RUN_ROOT}/generation/state"

generation_done="$(wc -l < "${RUN_ROOT}/generation/completed.tsv")"
generation_failed="$(wc -l < "${RUN_ROOT}/generation/failed.tsv")"
if [[ "${generation_failed}" -ne 0 || "${generation_done}" -ne "${NUM_GEN_JOBS}" ]]; then
  echo "[whole-block] generation failed: done=${generation_done}/${NUM_GEN_JOBS} failed=${generation_failed}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${PREPARE}" collect-roots \
  --output-base "${OUTPUT_BASE}" \
  --input-list "${INPUT_LIST}" \
  --whole-roots "${RUN_ROOT}/whole_block_leaf_folders.txt" \
  --plot-roots "${RUN_ROOT}/plot_leaf_folders.txt" \
  --merge-roots "${CURRENT_PLOTS}/current_roots.txt" \
  --report "${RUN_ROOT}/whole_block_validation.json"

mkdir -p \
  "${METRICS}/queues" \
  "${METRICS}/logs" \
  "${METRICS}/state" \
  "${METRICS}/task_summaries"
: > "${METRICS}/completed_tasks.tsv"
: > "${METRICS}/failed_tasks.tsv"

launch_metric_round pass1
launch_metric_round retry1
launch_metric_round retry2

"${PYTHON_BIN}" "${MANAGER}" verify-all \
  --all-roots "${RUN_ROOT}/whole_block_leaf_folders.txt" \
  --input-list "${INPUT_LIST}" \
  --output "${RUN_ROOT}/verification_final.json"

"${PYTHON_BIN}" "${SUMMARY}" \
  --input-txt "${RUN_ROOT}/whole_block_leaf_folders.txt" \
  --output-csv "${RUN_ROOT}/whole_block_metric_summary.csv" \
  --input-json-allowlist "${INPUT_LIST}"

mkdir -p "${WHOLE_PLOTS}" "${CURRENT_PLOTS}"
"${PYTHON_BIN}" "${PLOT}" \
  --input-txt "${RUN_ROOT}/plot_leaf_folders.txt" \
  --input-json-allowlist "${INPUT_LIST}" \
  --expected-cases 67 \
  --complete-only \
  --output-dir "${WHOLE_PLOTS}"

cp "${RUN_ROOT}/plot_leaf_folders.txt" "${CURRENT_PLOTS}/current_roots.txt"
"${PYTHON_BIN}" "${PLOT}" \
  --input-txt "${CURRENT_PLOTS}/current_roots.txt" \
  --input-json-allowlist "${INPUT_LIST}" \
  --expected-cases 67 \
  --complete-only \
  --output-dir "${CURRENT_PLOTS}"

"${PYTHON_BIN}" "${GALLERY}" \
  --result-root "${OUTPUT_BASE}" \
  --output-dir "${OUTPUT_BASE}/_gallery"

touch "${RUN_ROOT}/pipeline.complete"
echo "[whole-block] generation, all metrics, verification, plots, and gallery complete"
