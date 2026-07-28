#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_test5_st_phased_gpu012345_tmux.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${CONFIG:-${SCRIPT_DIR}/test5_st_phased_seed851.json}"
WORKER="${SCRIPT_DIR}/run_test5_st_phased_worker.py"
SESSION="${SESSION:-wan_test5_st_phased_seed851}"
OUTPUT_ROOT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["storage"]["output_root"])' "${CONFIG}")"
GPUS=(0 1 2 3 4 5)

"${PYTHON}" "${WORKER}" --config "${CONFIG}" --preflight
mkdir -p "${OUTPUT_ROOT}/worker_logs"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

for gpu in "${GPUS[@]}"; do
  name="gpu${gpu}"
  command="cd '${SCRIPT_DIR}' && PYTHONPATH='${SCRIPT_DIR}' '${PYTHON}' '${WORKER}' --config '${CONFIG}' --gpu '${gpu}' --worker-id '${name}' 2>&1 | tee -a '${OUTPUT_ROOT}/worker_logs/${name}.log'"
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION}" -n "${name}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${name}" "${command}"
  fi
done

echo "tmux=${SESSION}"
echo "output_root=${OUTPUT_ROOT}"
echo "status: tmux capture-pane -pt ${SESSION}:gpu0 | tail -40"
