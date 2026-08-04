#!/usr/bin/env bash
set -euo pipefail

SESSION="attention_lora_50seed_metrics"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
PREPARE="${DIFFTRACK}/AAA_my_test/prepare_attention_lora_seed_sweep_benchmark.py"
WORKER="${DIFFTRACK}/AAA_my_test/run_attention_lora_seed_metric_worker.sh"

tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}" || true
tmux new-session -d -s "${SESSION}" -n prepare \
  "while true; do ${PYTHON} ${PREPARE}; sleep 60; done; exec bash"
for gpu in 0 1 2 3; do
  tmux new-window -t "${SESSION}" -n "gpu${gpu}" \
    "bash ${WORKER} ${gpu}; exec bash"
done
tmux select-window -t "${SESSION}:prepare"
echo "started ${SESSION}: prepare + GPU 0-3"
