#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_pending_selected_qk_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${SCRIPT_DIR}/fulltoken_head_roles_test5_50seeds.json"
SNAPSHOT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/pending_selected_qk/pending_at_confirmation.json
GALLERY=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/multiseed
LOG_ROOT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/pending_selected_qk/logs
SESSION="${SESSION:-wan_pending_selected_qk}"

test -s "${CONFIG}"
test -s "${SNAPSHOT}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}"

worker_command() {
  local gpu="$1"
  printf \
    "%q %q --config %q --snapshot %q --gpu %q --worker-id %q 2>&1 | tee -a %q" \
    "${PYTHON}" \
    "${SCRIPT_DIR}/run_pending_selected_qk_worker.py" \
    "${CONFIG}" \
    "${SNAPSHOT}" \
    "${gpu}" \
    "gpu${gpu}" \
    "${LOG_ROOT}/worker_gpu${gpu}.log"
}

coordinator_command() {
  printf \
    "%q %q --config %q --snapshot %q --gallery-dir %q --poll-seconds 300 2>&1 | tee -a %q" \
    "${PYTHON}" \
    "${SCRIPT_DIR}/coordinate_pending_selected_qk.py" \
    "${CONFIG}" \
    "${SNAPSHOT}" \
    "${GALLERY}" \
    "${LOG_ROOT}/coordinator.log"
}

tmux new-session -d -s "${SESSION}" -n coordinator "$(coordinator_command)"
for gpu in 0 1 2 3 4 5; do
  tmux new-window -t "${SESSION}" -n "gpu${gpu}" "$(worker_command "${gpu}")"
done

echo "started tmux session: ${SESSION}"
echo "pending QK jobs: 70 model-seed tasks frozen in ${SNAPSHOT}"
echo "gallery: ${GALLERY}"
echo "attach: tmux attach -t ${SESSION}"
