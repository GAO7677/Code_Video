#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_incremental_remaining_metrics_gpu0123456_tmux.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PREPARE="${SCRIPT_DIR}/prepare_incremental_remaining_metrics.py"
WORKER="${SCRIPT_DIR}/run_remaining_blocks_queue_worker.sh"
SOURCE_RUN=/data/gaoya/AAA_test_video/0623/test/v2v_wan/_remaining_blocks_pipeline
INPUT_LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
SESSION=remaining_blocks_incremental_metrics_gpu0123456
RUN_ROOT="${SOURCE_RUN}/incremental_metrics_$(date -u +%Y%m%dT%H%M%SZ)"
GPUS=(0 1 2 3 4 5 6)
CPU_WORKERS_PER_GPU=2

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
"${PYTHON_BIN}" "${PREPARE}" \
  --completed "${SOURCE_RUN}/generation/completed.tsv" \
  --input-list "${INPUT_LIST}" \
  --run-root "${RUN_ROOT}"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do complete=\$(find '${RUN_ROOT}/state' -maxdepth 1 -type f -name '*.complete' | wc -l); failed=\$(wc -l < '${RUN_ROOT}/failed_tasks.tsv'); printf '[incremental] workers=%s/21 completed_tasks=%s failed_tasks=%s\\n' \"\$complete\" \"\$(wc -l < '${RUN_ROOT}/completed_tasks.tsv')\" \"\$failed\"; [ \"\$complete\" -eq 21 ] && break; sleep 30; done; touch '${RUN_ROOT}/complete'; exec bash"

for gpu in "${GPUS[@]}"; do
  for index in $(seq 0 $((CPU_WORKERS_PER_GPU - 1))); do
    name="cpu_g${gpu}_${index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' '${gpu}' cpu '${name}' '${RUN_ROOT}' '${INPUT_LIST}'; exec bash"
  done
  name="gpu_g${gpu}"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${WORKER}' '${gpu}' gpu '${name}' '${RUN_ROOT}' '${INPUT_LIST}'; exec bash"
done

tmux select-window -t "${SESSION}:coordinator"
echo "session=${SESSION}"
echo "run_root=${RUN_ROOT}"
echo "cpu_workers=14 gpu_workers=7"
