#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_remaining_recovery_gpu012_gen_gpu3_metrics_tmux.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
OUTPUT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v_wan
SOURCE_RUN="${OUTPUT_BASE}/_remaining_blocks_pipeline"
SOURCE_INCREMENTAL="${SOURCE_RUN}/incremental_metrics_20260726T041711Z"
INPUT_LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
SESSION=remaining_blocks_recovery_gen012_metric3
RUN_ROOT="${SOURCE_RUN}/recovery_$(date -u +%Y%m%dT%H%M%SZ)"
GEN_WORKER="${SCRIPT_DIR}/run_remaining_blocks_generation_worker.sh"
METRIC_WORKER="${SCRIPT_DIR}/run_remaining_blocks_queue_worker.sh"
PREPARE_GEN="${SCRIPT_DIR}/prepare_remaining_generation_recovery.py"
MANAGER="${SCRIPT_DIR}/manage_remaining_block_pipeline.py"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p \
  "${RUN_ROOT}/generation/logs" \
  "${RUN_ROOT}/generation/state" \
  "${RUN_ROOT}/generation/validations" \
  "${RUN_ROOT}/metrics/queues" \
  "${RUN_ROOT}/metrics/logs" \
  "${RUN_ROOT}/metrics/state" \
  "${RUN_ROOT}/metrics/task_summaries"

"${PYTHON_BIN}" "${PREPARE_GEN}" \
  --manifest "${SOURCE_RUN}/generation_manifest.json" \
  --input-list "${INPUT_LIST}" \
  --queue "${RUN_ROOT}/generation/queue.tsv" \
  --report "${RUN_ROOT}/generation_manifest.json"
printf '1\n' > "${RUN_ROOT}/generation/cursor"
: > "${RUN_ROOT}/generation/queue.lock"
: > "${RUN_ROOT}/generation/completed.tsv"
: > "${RUN_ROOT}/generation/failed.tsv"

"${PYTHON_BIN}" "${MANAGER}" build-retry \
  --all-roots "${SOURCE_INCREMENTAL}/completed_roots.txt" \
  --input-list "${INPUT_LIST}" \
  --queue "${RUN_ROOT}/metrics/queues/retry.tsv" \
  --report "${RUN_ROOT}/metric_retry_manifest.json"
printf '1\n' > "${RUN_ROOT}/metrics/queues/retry.cursor"
: > "${RUN_ROOT}/metrics/queues/retry.lock"
: > "${RUN_ROOT}/metrics/completed_tasks.tsv"
: > "${RUN_ROOT}/metrics/failed_tasks.tsv"

num_generation="$(wc -l < "${RUN_ROOT}/generation/queue.tsv")"
num_metrics="$(wc -l < "${RUN_ROOT}/metrics/queues/retry.tsv")"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do gen=\$(find '${RUN_ROOT}/generation/state' -maxdepth 1 -name 'recovery_gen_g*.complete' | wc -l); met=\$(find '${RUN_ROOT}/metrics/state' -maxdepth 1 -name 'recovery_metric_g3.complete' | wc -l); printf '[recovery] generation_workers=%s/3 generation_done=%s/${num_generation} generation_failed=%s metric_workers=%s/1 metrics_done=%s/${num_metrics} metrics_failed=%s\\n' \"\$gen\" \"\$(wc -l < '${RUN_ROOT}/generation/completed.tsv')\" \"\$(wc -l < '${RUN_ROOT}/generation/failed.tsv')\" \"\$met\" \"\$(wc -l < '${RUN_ROOT}/metrics/completed_tasks.tsv')\" \"\$(wc -l < '${RUN_ROOT}/metrics/failed_tasks.tsv')\"; [ \"\$gen\" -eq 3 ] && [ \"\$met\" -eq 1 ] && break; sleep 30; done; touch '${RUN_ROOT}/recovery.complete'; exec bash"

for gpu in 0 1 2; do
  name="recovery_gen_g${gpu}"
  tmux new-window -d -t "${SESSION}" -n "${name}" \
    "bash '${GEN_WORKER}' '${gpu}' '${name}' '${RUN_ROOT}' '${OUTPUT_BASE}' '${INPUT_LIST}'; exec bash"
done

tmux new-window -d -t "${SESSION}" -n recovery_metric_g3 \
  "bash '${METRIC_WORKER}' 3 retry recovery_metric_g3 '${RUN_ROOT}/metrics' '${INPUT_LIST}'; exec bash"

tmux select-window -t "${SESSION}:coordinator"
echo "session=${SESSION}"
echo "run_root=${RUN_ROOT}"
echo "generation_pending=${num_generation}"
echo "metric_retry_pending=${num_metrics}"
