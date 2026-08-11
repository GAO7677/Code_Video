#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/physrvg_verified_remote118_gpu67.env"

echo "[sync] adapter and exact BPP inputs"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_ADAPTER_DIR}' '${REMOTE_INPUT_ROOT}' '${REMOTE_OUTPUT_ROOT}' '${REMOTE_LOG_ROOT}'"
rsync -a "${LOCAL_ADAPTER_DIR}/" "${REMOTE_HOST}:${REMOTE_ADAPTER_DIR}/"
rsync -a "${LOCAL_INPUT_ROOT}/" "${REMOTE_HOST}:${REMOTE_INPUT_ROOT}/"

if ssh "${REMOTE_HOST}" "tmux has-session -t '${TMUX_SESSION}'" 2>/dev/null; then
  echo "[error] tmux session already exists: ${TMUX_SESSION}" >&2
  exit 1
fi

ssh "${REMOTE_HOST}" \
  "tmux new-session -d -s '${TMUX_SESSION}' \"bash '${REMOTE_ADAPTER_DIR}/run_physrvg_verified_gpu67_remote_worker.sh' 2>&1 | tee '${REMOTE_LOG_ROOT}/${RUN_NAME}_pipeline.log'\""

echo "started host=${REMOTE_HOST} tmux=${TMUX_SESSION} gpu=6,7 run=${RUN_NAME}"
echo "attach: ssh -t ${REMOTE_HOST} tmux attach -t ${TMUX_SESSION}"
echo "log: ${REMOTE_LOG_ROOT}/${RUN_NAME}_pipeline.log"
