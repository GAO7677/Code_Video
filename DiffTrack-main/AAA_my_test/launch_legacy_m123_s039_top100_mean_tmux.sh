#!/usr/bin/env bash
set -euo pipefail

SESSION=${1:-m123_s039_top100_mean_capture}
RUNNER=/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_m123_s039_top100_mean_gpu.sh
GPUS=(2 3)
NUM_WORKERS=${#GPUS[@]}

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

worker_command() {
  local gpu=$1 worker=$2
  printf '%s' "while true; do used=\$(nvidia-smi --id=${gpu} --query-gpu=memory.used --format=csv,noheader,nounits | xargs); if [[ \"\${used}\" =~ ^[0-9]+$ ]] && (( used < 1024 )); then break; fi; echo \"[\$(date -u +%FT%TZ)] waiting GPU${gpu}: \${used} MiB\"; sleep 30; done; exec bash ${RUNNER} ${gpu} ${worker} ${NUM_WORKERS}"
}

tmux new-session -d -s "${SESSION}" -n gpu2 "bash -lc '$(worker_command 2 0)'"
for worker in 1; do
  gpu=${GPUS[$worker]}
  tmux new-window -d -t "${SESSION}" -n "gpu${gpu}" \
    "bash -lc '$(worker_command "${gpu}" "${worker}")'"
done

echo "started ${SESSION}: GPUs 2,3 wait until fully free, then run 54 tasks each"
