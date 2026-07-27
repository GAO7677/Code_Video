#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physrvg_whole_block_priority_gpu012_tmux.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
OUTPUT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v_wan
PIPELINE_ROOT="${OUTPUT_BASE}/_remaining_blocks_pipeline"
INPUT_LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
SESSION=physrvg_whole_block_priority_gpu012
RUN_ROOT="${PIPELINE_ROOT}/physrvg_whole_block_priority_$(date -u +%Y%m%dT%H%M%SZ)"
PREPARE_GEN="${SCRIPT_DIR}/prepare_remaining_generation_recovery.py"
GEN_WORKER="${SCRIPT_DIR}/run_remaining_blocks_generation_worker.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p \
  "${RUN_ROOT}/generation/logs" \
  "${RUN_ROOT}/generation/state" \
  "${RUN_ROOT}/generation/validations"

"${PYTHON_BIN}" "${PREPARE_GEN}" \
  --manifest "${PIPELINE_ROOT}/generation_manifest.json" \
  --input-list "${INPUT_LIST}" \
  --queue "${RUN_ROOT}/generation/queue.unordered.tsv" \
  --report "${RUN_ROOT}/generation_manifest.json"

awk -F $'\t' '$2 == "physrvg" && $3 == "whole_block"' \
  "${RUN_ROOT}/generation/queue.unordered.tsv" \
  > "${RUN_ROOT}/generation/queue.priority.tsv"
awk -F $'\t' '!($2 == "physrvg" && $3 == "whole_block")' \
  "${RUN_ROOT}/generation/queue.unordered.tsv" \
  > "${RUN_ROOT}/generation/queue.remaining.tsv"
awk '{ print }' \
  "${RUN_ROOT}/generation/queue.priority.tsv" \
  "${RUN_ROOT}/generation/queue.remaining.tsv" \
  > "${RUN_ROOT}/generation/queue.tsv"

priority_jobs="$(wc -l < "${RUN_ROOT}/generation/queue.priority.tsv")"
total_jobs="$(wc -l < "${RUN_ROOT}/generation/queue.tsv")"
if [[ "${priority_jobs}" -ne 24 ]]; then
  echo "Expected 24 pending PhysRVG whole_block jobs, found ${priority_jobs}" >&2
  exit 2
fi

printf '1\n' > "${RUN_ROOT}/generation/cursor"
: > "${RUN_ROOT}/generation/queue.lock"
: > "${RUN_ROOT}/generation/completed.tsv"
: > "${RUN_ROOT}/generation/failed.tsv"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do workers=\$(find '${RUN_ROOT}/generation/state' -maxdepth 1 -name 'priority_gen_g*.complete' | wc -l); done_count=\$(wc -l < '${RUN_ROOT}/generation/completed.tsv'); failed_count=\$(wc -l < '${RUN_ROOT}/generation/failed.tsv'); priority_done=\$(awk -F '\\t' '\$2 == \"physrvg\" && \$3 == \"whole_block\" {count++} END {print count + 0}' '${RUN_ROOT}/generation/completed.tsv'); printf '[priority] workers=%s/3 priority_whole_block=%s/${priority_jobs} total_done=%s/${total_jobs} failed=%s\\n' \"\$workers\" \"\$priority_done\" \"\$done_count\" \"\$failed_count\"; [ \"\$workers\" -eq 3 ] && break; sleep 30; done; touch '${RUN_ROOT}/complete'; exec bash"

for gpu in 0 1 2; do
  worker="priority_gen_g${gpu}"
  tmux new-window -d -t "${SESSION}" -n "${worker}" \
    "bash '${GEN_WORKER}' '${gpu}' '${worker}' '${RUN_ROOT}' '${OUTPUT_BASE}' '${INPUT_LIST}'; exec bash"
done

tmux select-window -t "${SESSION}:coordinator"
echo "session=${SESSION}"
echo "run_root=${RUN_ROOT}"
echo "priority_physrvg_whole_block=${priority_jobs}"
echo "total_pending=${total_jobs}"
