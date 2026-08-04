#!/usr/bin/env bash
set -euo pipefail

GEN_SESSION="attention_lora_multicase_queue"
METRIC_SESSION="attention_lora_multicase_metrics"
HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"

if ! tmux has-session -t "${GEN_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${GEN_SESSION}" -n queue \
    "bash ${HERE}/run_attention_lora_multicase_queue.sh; exec bash"
fi
if ! tmux has-session -t "${METRIC_SESSION}" 2>/dev/null; then
  for gpu in 0 1 2 3; do
    command="bash ${HERE}/run_attention_lora_multicase_metric_monitor.sh ${gpu}; exec bash"
    if [[ "${gpu}" == 0 ]]; then
      tmux new-session -d -s "${METRIC_SESSION}" -n "gpu${gpu}" "${command}"
    else
      tmux new-window -t "${METRIC_SESSION}" -n "gpu${gpu}" "${command}"
    fi
  done
fi
echo "started ${GEN_SESSION} and ${METRIC_SESSION}; GPU 6/7 are excluded"
