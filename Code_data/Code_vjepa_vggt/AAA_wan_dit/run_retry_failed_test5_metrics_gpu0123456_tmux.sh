#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_retry_failed_test5_metrics_gpu0123456_tmux.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER="${SCRIPT_DIR}/run_bench_v2v_wan_queue_worker.sh"
BUILDER="${SCRIPT_DIR}/build_failed_metric_retry_queues.py"
COORDINATOR="${SCRIPT_DIR}/run_retry_failed_test5_metrics_coordinator.sh"
OUTPUT_BASE=/data/gaoya/agent-data/outputs/wan_dit_ablation/test5_first5
PIPELINE_ROOT="${OUTPUT_BASE}/_pipeline"
SOURCE_SUMMARY_DIR="${PIPELINE_ROOT}/metrics/task_summaries"
INPUT_LIST="${PIPELINE_ROOT}/input_first5_unique.txt"
LEAF_LIST="${PIPELINE_ROOT}/leaf_folders.txt"
SESSION="${SESSION:-test5_retry_failed_metrics_20260725}"
RUN_ROOT="${RUN_ROOT:-${PIPELINE_ROOT}/metric_retry_failed_20260725}"
GPUS=(0 1 2 3 4 5 6)
COSMOS_GPUS=(0 1 2 3 4 5)

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "retry run root already exists: ${RUN_ROOT}" >&2
  exit 1
fi
if [[ ! -s "${INPUT_LIST}" || ! -s "${LEAF_LIST}" ]]; then
  echo "missing input or leaf list" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/state" "${RUN_ROOT}/task_summaries"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"

"${PYTHON_BIN}" "${BUILDER}" \
  --summary-dir "${SOURCE_SUMMARY_DIR}" \
  --output-dir "${RUN_ROOT}/queues" \
  --report "${RUN_ROOT}/retry_manifest.json"

for kind in videophy2 cosmos gpu_common; do
  printf '1\n' > "${RUN_ROOT}/queues/${kind}.cursor"
  : > "${RUN_ROOT}/queues/${kind}.lock"
done

VP_TASKS="$(wc -l < "${RUN_ROOT}/queues/videophy2.tsv")"
COSMOS_TASKS="$(wc -l < "${RUN_ROOT}/queues/cosmos.tsv")"
COMMON_TASKS="$(wc -l < "${RUN_ROOT}/queues/gpu_common.tsv")"
TOTAL_TASKS=$((VP_TASKS + COSMOS_TASKS + COMMON_TASKS))
if [[ "${TOTAL_TASKS}" -eq 0 ]]; then
  echo "No failed metric tasks require retry."
  exit 0
fi

tmux new-session -d -s "${SESSION}" -n coordinator \
  "bash '${COORDINATOR}' '${RUN_ROOT}' '${#GPUS[@]}' '${#COSMOS_GPUS[@]}' '${#GPUS[@]}' '${INPUT_LIST}' '${LEAF_LIST}' '${OUTPUT_BASE}' '${PIPELINE_ROOT}'; exec bash"

for gpu in "${GPUS[@]}"; do
  name="vp_g${gpu}"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "while [[ ! -f '${RUN_ROOT}/videophy2.ready' ]]; do sleep 5; done; bash '${WORKER}' '${gpu}' videophy2 '${name}' '${RUN_ROOT}' '${INPUT_LIST}'; exec bash"
done

for gpu in "${COSMOS_GPUS[@]}"; do
  name="cosmos_g${gpu}"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "while [[ ! -f '${RUN_ROOT}/cosmos.ready' ]]; do sleep 5; done; bash '${WORKER}' '${gpu}' cosmos '${name}' '${RUN_ROOT}' '${INPUT_LIST}'; exec bash"
done

for gpu in "${GPUS[@]}"; do
  name="common_g${gpu}"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "while [[ ! -f '${RUN_ROOT}/gpu_common.ready' ]]; do sleep 5; done; bash '${WORKER}' '${gpu}' gpu_common '${name}' '${RUN_ROOT}' '${INPUT_LIST}'; exec bash"
done

tmux select-window -t "${SESSION}:coordinator"
echo "session=${SESSION}"
echo "run_root=${RUN_ROOT}"
echo "retry_tasks=${TOTAL_TASKS} (videophy2=${VP_TASKS}, cosmos=${COSMOS_TASKS}, gpu_common=${COMMON_TASKS})"
echo "concurrency=one GPU-heavy worker per GPU; stages run sequentially"
