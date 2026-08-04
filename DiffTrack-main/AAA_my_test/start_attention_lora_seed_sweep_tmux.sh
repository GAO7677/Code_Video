#!/usr/bin/env bash
set -euo pipefail

SESSION="attention_lora_50seeds_case001460"
SCRIPT="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_attention_lora_seed_sweep_gpu.sh"

tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
}
for gpu in 0 1 2 3 4 5 6 7; do
  if [[ "${gpu}" == 0 ]]; then
    tmux new-session -d -s "${SESSION}" -n "gpu${gpu}" "while ! bash ${SCRIPT} ${gpu} 8; do echo RETRY_GPU_${gpu}; sleep 300; done; exec bash"
  else
    tmux new-window -t "${SESSION}" -n "gpu${gpu}" "while ! bash ${SCRIPT} ${gpu} 8; do echo RETRY_GPU_${gpu}; sleep 300; done; exec bash"
  fi
done
tmux select-window -t "${SESSION}:gpu0"
echo "started ${SESSION} on GPU 0-7"
