#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_common22_public_head_ablation_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${CONFIG:-${SCRIPT_DIR}/common22_public_head_ablation.json}"
SESSION="${SESSION:-wan_common22_public_head_ablation}"
OUTPUT_ROOT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["storage"]["output_root"])' "${CONFIG}")"
GPUS=(0 1 2 3 4 5 6)

mkdir -p "${OUTPUT_ROOT}/worker_logs"
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
}
for gpu in "${GPUS[@]}"; do
  name="gpu${gpu}"
  command="cd '${SCRIPT_DIR}' && PYTHONPATH='${SCRIPT_DIR}' '${PYTHON}' '${SCRIPT_DIR}/run_common22_public_head_ablation_worker.py' --config '${CONFIG}' --gpu '${gpu}' --worker-id '${name}' 2>&1 | tee -a '${OUTPUT_ROOT}/worker_logs/${name}.log'"
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION}" -n "${name}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${name}" "${command}"
  fi
done
tmux new-window -t "${SESSION}" -n gallery \
  "cd '${SCRIPT_DIR}' && CONFIG='${CONFIG}' bash '${SCRIPT_DIR}/watch_common22_public_head_gallery.sh' 2>&1 | tee -a '${OUTPUT_ROOT}/worker_logs/gallery.log'"

echo "tmux=${SESSION}"
echo "gallery=http://127.0.0.1:8944/multiseed/"
