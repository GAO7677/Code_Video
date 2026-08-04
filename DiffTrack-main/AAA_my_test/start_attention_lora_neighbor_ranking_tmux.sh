#!/usr/bin/env bash
set -euo pipefail

HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
GEN="attention_lora_neighbor_ranking_50seeds"
METRICS="attention_lora_neighbor_ranking_metrics"

if ! tmux has-session -t "${GEN}" 2>/dev/null; then
  tmux new-session -d -s "${GEN}" -n queue \
    "bash ${HERE}/run_attention_lora_neighbor_ranking_queue.sh; exec bash"
fi
if ! tmux has-session -t "${METRICS}" 2>/dev/null; then
  for gpu in 0 1 2 3; do
    command="bash ${HERE}/run_attention_lora_neighbor_ranking_metric_worker.sh ${gpu}; exec bash"
    if [[ "${gpu}" == 0 ]]; then
      tmux new-session -d -s "${METRICS}" -n "gpu${gpu}" "${command}"
    else
      tmux new-window -t "${METRICS}" -n "gpu${gpu}" "${command}"
    fi
  done
fi
echo "started ${GEN} and ${METRICS}; GPU 6/7 excluded"
