#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_three_train_checkpoint_watch_tmux.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/xssc_lora_three_train_watch_config.json}"
if [[ "${CONFIG}" == "${SCRIPT_DIR}/xssc_lora_three_train_watch_config.json" ]]; then
  CONFIG="${SCRIPT_DIR}/xssc_lora_three_train_watch_config_with_t_head.json"
fi
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WATCHER="${SCRIPT_DIR}/xssc_lora_checkpoint_watch.py"
PHYSICIQ_WATCHER="${SCRIPT_DIR}/xssc_lora_physiciq_watch.py"
REFRESHER="${SCRIPT_DIR}/xssc_lora_dashboard_refresh_loop.py"
SESSION="${SESSION:-wan_train}"

if [[ ! -s "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi

declare -A COMMANDS=(
  [three_ckpt_watch]="${PYTHON} ${WATCHER} --config ${CONFIG} --mode inference"
  [three_test5_cpu]="${PYTHON} ${WATCHER} --config ${CONFIG} --mode metrics --kind cpu"
  [three_test5_gpu]="${PYTHON} ${WATCHER} --config ${CONFIG} --mode metrics --kind gpu"
  [three_phys_watch]="${PYTHON} ${PHYSICIQ_WATCHER} --config ${CONFIG} --mode inference"
  [three_phys_cpu]="${PYTHON} ${PHYSICIQ_WATCHER} --config ${CONFIG} --mode metrics --kind cpu"
  [three_phys_gpu]="${PYTHON} ${PHYSICIQ_WATCHER} --config ${CONFIG} --mode metrics --kind gpu"
  [three_dashboard]="${PYTHON} ${REFRESHER} --config ${CONFIG} --interval 60"
)

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n shell
fi
for window_name in "${!COMMANDS[@]}"; do
  if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${window_name}"; then
    echo "tmux window already exists: ${SESSION}:${window_name}" >&2
    exit 1
  fi
done

"${PYTHON}" "${WATCHER}" --config "${CONFIG}" --mode bootstrap
for window_name in \
  three_ckpt_watch three_test5_cpu three_test5_gpu \
  three_phys_watch three_phys_cpu three_phys_gpu three_dashboard; do
  command="${COMMANDS[${window_name}]}"
  tmux new-window -t "${SESSION}" -n "${window_name}" "${command}; exec bash"
done

echo "watcher session: ${SESSION}"
echo "config: ${CONFIG}"
echo "overview: http://127.0.0.1:8951/"
echo "gateway root: /data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_visualizations"
echo "dashboard refresh interval: 60s"
echo "GPU inference/metrics device: $(/home/gaoya/miniconda3/envs/wan-cu128/bin/python - <<'PY'
import json
import os
config_path = os.environ['CONFIG']
print(json.load(open(config_path, 'r', encoding='utf-8'))['runtime']['gpu_id'])
PY
)"
