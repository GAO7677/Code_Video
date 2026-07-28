#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_stc_steps00_05_handoff_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-wan_stc_steps00_05_handoff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_common22_public_head_ablation_case025}"
LOG_ROOT="${OUTPUT_ROOT}/worker_logs/stc_steps00_05_multiseed"
mkdir -p "${LOG_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

for gpu in 0 1 2 3 4 5 6; do
  name="gpu${gpu}"
  command="cd '${SCRIPT_DIR}' && GPU='${gpu}' bash '${SCRIPT_DIR}/wait_then_run_stc_steps00_05_gpu.sh' 2>&1 | tee -a '${LOG_ROOT}/${name}.log'"
  if (( gpu == 0 )); then
    tmux new-session -d -s "${SESSION}" -n "${name}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${name}" "${command}"
  fi
done

echo "started tmux session ${SESSION}"
tmux list-windows -t "${SESSION}"
