#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_all_whole_block_gen_metrics_gpu012_tmux.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
OUTPUT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v_wan
PIPELINE_ROOT="${OUTPUT_BASE}/_remaining_blocks_pipeline"
INPUT_LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
SESSION=all_whole_block_gen_metrics_gpu012
RUN_ROOT="${PIPELINE_ROOT}/all_whole_block_$(date -u +%Y%m%dT%H%M%SZ)"
PREPARE="${SCRIPT_DIR}/prepare_whole_block_pipeline.py"
GEN_WORKER="${SCRIPT_DIR}/run_remaining_blocks_generation_worker.sh"
COORDINATOR="${SCRIPT_DIR}/run_all_whole_block_coordinator.sh"
NUM_GEN_WORKERS=3

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p \
  "${RUN_ROOT}/generation/logs" \
  "${RUN_ROOT}/generation/state" \
  "${RUN_ROOT}/generation/validations"

"${PYTHON_BIN}" "${PREPARE}" build-generation \
  --output-base "${OUTPUT_BASE}" \
  --input-list "${INPUT_LIST}" \
  --queue "${RUN_ROOT}/generation/queue.tsv" \
  --report "${RUN_ROOT}/generation_manifest.json"

num_generation="$(wc -l < "${RUN_ROOT}/generation/queue.tsv")"
printf '1\n' > "${RUN_ROOT}/generation/cursor"
: > "${RUN_ROOT}/generation/queue.lock"
: > "${RUN_ROOT}/generation/completed.tsv"
: > "${RUN_ROOT}/generation/failed.tsv"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "bash '${COORDINATOR}' '${RUN_ROOT}' '${OUTPUT_BASE}' '${INPUT_LIST}' '${SESSION}' '${num_generation}' '${NUM_GEN_WORKERS}'; exec bash"

for gpu in 0 1 2; do
  name="whole_gen_g${gpu}"
  tmux new-window -d -t "${SESSION}" -n "${name}" \
    "bash '${GEN_WORKER}' '${gpu}' '${name}' '${RUN_ROOT}' '${OUTPUT_BASE}' '${INPUT_LIST}'; exec bash"
done

tmux select-window -t "${SESSION}:coordinator"
echo "session=${SESSION}"
echo "run_root=${RUN_ROOT}"
echo "whole_block_pending=${num_generation}"
echo "gpu_ids=0,1,2"
