#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_xssc_lora_checkpoint_watch_tmux.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/xssc_lora_checkpoint_watch_config.json}"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WATCHER="${SCRIPT_DIR}/xssc_lora_checkpoint_watch.py"
PHYSICIQ_WATCHER="${SCRIPT_DIR}/xssc_lora_physiciq_watch.py"
SESSION="${SESSION:-wan_train}"
PORT="${PORT:-8844}"
HUB_ROOT=/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub
STABLE_ROOT=/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_visualizations

if [[ ! -s "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi
if ss -ltn "sport = :${PORT}" | tail -n +2 | grep -q .; then
  echo "Port ${PORT} is already in use; stop the existing overview service first." >&2
  exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  for window_name in \
    overview checkpoint_watch metrics_cpu metrics_gpu \
    physiciq_watch physiciq_metrics_cpu physiciq_metrics_gpu; do
    if tmux list-windows -t "${SESSION}" -F '#{window_name}' |
      grep -Fxq "${window_name}"; then
      echo "tmux window already exists: ${SESSION}:${window_name}" >&2
      exit 1
    fi
  done
fi

"${PYTHON}" "${WATCHER}" --config "${CONFIG}" --mode bootstrap
ln -sfn "${HUB_ROOT}" "${STABLE_ROOT}"

server_command=(
  "${PYTHON}" -m http.server "${PORT}"
  --bind 127.0.0.1
  --directory "${STABLE_ROOT}"
)
inference_command=(
  "${PYTHON}" "${WATCHER}" --config "${CONFIG}" --mode inference
)
cpu_metric_command=(
  "${PYTHON}" "${WATCHER}" --config "${CONFIG}" --mode metrics --kind cpu
)
gpu_metric_command=(
  "${PYTHON}" "${WATCHER}" --config "${CONFIG}" --mode metrics --kind gpu
)
physiciq_inference_command=(
  "${PYTHON}" "${PHYSICIQ_WATCHER}" --config "${CONFIG}" --mode inference
)
physiciq_cpu_metric_command=(
  "${PYTHON}" "${PHYSICIQ_WATCHER}" --config "${CONFIG}" --mode metrics --kind cpu
)
physiciq_gpu_metric_command=(
  "${PYTHON}" "${PHYSICIQ_WATCHER}" --config "${CONFIG}" --mode metrics --kind gpu
)

printf -v server_shell '%q ' "${server_command[@]}"
printf -v inference_shell '%q ' "${inference_command[@]}"
printf -v cpu_metric_shell '%q ' "${cpu_metric_command[@]}"
printf -v gpu_metric_shell '%q ' "${gpu_metric_command[@]}"
printf -v physiciq_inference_shell '%q ' "${physiciq_inference_command[@]}"
printf -v physiciq_cpu_metric_shell '%q ' "${physiciq_cpu_metric_command[@]}"
printf -v physiciq_gpu_metric_shell '%q ' "${physiciq_gpu_metric_command[@]}"

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n overview \
    "${server_shell}; exec bash"
else
  tmux new-window -t "${SESSION}" -n overview \
    "${server_shell}; exec bash"
fi
tmux new-window -t "${SESSION}" -n checkpoint_watch \
  "${inference_shell}; exec bash"
tmux new-window -t "${SESSION}" -n metrics_cpu \
  "${cpu_metric_shell}; exec bash"
tmux new-window -t "${SESSION}" -n metrics_gpu \
  "${gpu_metric_shell}; exec bash"
tmux new-window -t "${SESSION}" -n physiciq_watch \
  "${physiciq_inference_shell}; exec bash"
tmux new-window -t "${SESSION}" -n physiciq_metrics_cpu \
  "${physiciq_cpu_metric_shell}; exec bash"
tmux new-window -t "${SESSION}" -n physiciq_metrics_gpu \
  "${physiciq_gpu_metric_shell}; exec bash"
tmux select-window -t "${SESSION}:overview"

echo "tmux session: ${SESSION}"
echo "overview: http://127.0.0.1:${PORT}/"
echo "watch root: /data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_checkpoint_watch"
echo "foreground server command: ${server_shell}"
