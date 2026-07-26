#!/usr/bin/env bash
set -euo pipefail

# SESSION=wan_paired_query_50seeds bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_paired_query_50seeds_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-wan_paired_query_50seeds}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/paired_query_head_stability_test5_50seeds.json}"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

test -s "${CONFIG}"
mkdir -p "${OUTPUT_ROOT}/worker_logs"

worker_command() {
  local index="$1"
  local gpu="$2"
  printf \
    "%q %q --config %q --gpu %q --worker-index %q --worker-count 3 2>&1 | tee %q" \
    "${PYTHON}" \
    "${SCRIPT_DIR}/run_paired_query_50seeds_worker.py" \
    "${CONFIG}" \
    "${gpu}" \
    "${index}" \
    "${OUTPUT_ROOT}/worker_logs/worker${index}_gpu${gpu}.log"
}

tmux new-session -d -s "${SESSION}" -n wan_lora \
  "$(worker_command 0 4)"
tmux new-window -t "${SESSION}" -n xssc \
  "$(worker_command 1 5)"
tmux new-window -t "${SESSION}" -n physrvg \
  "$(worker_command 2 6)"

echo "started ${SESSION}: wan_lora=GPU4, xssc=GPU5, physrvg=GPU6"
echo "output: ${OUTPUT_ROOT}"
echo "attach: tmux attach -t ${SESSION}"
