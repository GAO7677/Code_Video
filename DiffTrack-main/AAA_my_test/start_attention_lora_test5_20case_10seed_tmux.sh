#!/usr/bin/env bash
set -euo pipefail

SESSION="attention_lora_test5_20case_10seed"
HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
PREPARE="${HERE}/prepare_attention_lora_test5_20case_10seed.py"
WORKER="${HERE}/run_attention_lora_test5_20case_10seed_gpu.sh"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"

"${PYTHON}" "${PREPARE}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n gpu2 \
  "bash '${WORKER}' 2 0; exec bash"
tmux new-window -t "${SESSION}" -n gpu3 \
  "bash '${WORKER}' 3 1; exec bash"
tmux select-window -t "${SESSION}:gpu2"
echo "started ${SESSION}; GPU2/3 workers are waiting for stable idle memory"
