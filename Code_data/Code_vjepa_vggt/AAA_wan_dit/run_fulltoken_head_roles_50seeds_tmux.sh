#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_fulltoken_head_roles_50seeds_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/fulltoken_head_roles_test5_50seeds.json}"
SESSION="${SESSION:-wan_fulltoken_head_roles_50seeds}"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds

test -s "${CONFIG}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/worker_logs" "${OUTPUT_ROOT}/finalizer_logs"

worker_command() {
  local gpu="$1"
  printf \
    "%q %q --config %q --gpu %q --worker-id %q 2>&1 | tee -a %q" \
    "${PYTHON}" \
    "${SCRIPT_DIR}/run_fulltoken_head_roles_worker.py" \
    "${CONFIG}" \
    "${gpu}" \
    "gpu${gpu}" \
    "${OUTPUT_ROOT}/worker_logs/gpu${gpu}.log"
}

finalizer_command() {
  printf \
    "%q %q --config %q --poll-seconds 300 2>&1 | tee -a %q" \
    "${PYTHON}" \
    "${SCRIPT_DIR}/finalize_fulltoken_head_roles_batch.py" \
    "${CONFIG}" \
    "${OUTPUT_ROOT}/finalizer_logs/watcher.log"
}

tmux new-session -d -s "${SESSION}" -n gpu0 "$(worker_command 0)"
for gpu in 1 2 3 4 5; do
  tmux new-window -t "${SESSION}" -n "gpu${gpu}" "$(worker_command "${gpu}")"
done
tmux new-window -t "${SESSION}" -n finalizer "$(finalizer_command)"

echo "started tmux session: ${SESSION}"
echo "workers: GPU0,1,2,3,4,5 (one worker per GPU)"
echo "output: ${OUTPUT_ROOT}"
echo "attach: tmux attach -t ${SESSION}"
