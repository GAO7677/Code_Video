#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_head_role_s_feature_union_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${SCRIPT_DIR}/head_role_s_feature_union_pilot.json"
SOURCE_MANIFEST=/data/gaoya/agent-data/outputs/wan_dit_s_feature_split/configs/s_feature_split_subsets.json
UNION_MANIFEST=/data/gaoya/agent-data/outputs/wan_dit_s_feature_union/configs/s_feature_union_subset.json
SESSION=wan_s_feature_union
GPUS=(2 3 5)

"${PYTHON}" "${SCRIPT_DIR}/build_s_feature_union_subset.py" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --output "${UNION_MANIFEST}"
"${PYTHON}" "${SCRIPT_DIR}/run_head_role_dose_control_pilot_worker.py" \
  --config "${CONFIG}" --preflight

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n shell
fi

for gpu in "${GPUS[@]}"; do
  name="union_g${gpu}"
  if tmux list-windows -t "${SESSION}" -F '#W' | rg -Fxq "${name}"; then
    echo "window already exists: ${SESSION}:${name}"
    continue
  fi
  command=(
    "${PYTHON}"
    "${SCRIPT_DIR}/run_head_role_dose_control_pilot_worker.py"
    --config "${CONFIG}"
    --gpu "${gpu}"
    --worker-id "${name}"
  )
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-window -d -t "${SESSION}" -n "${name}" \
    "${shell_command}; exec bash"
done

watcher=(
  "${PYTHON}"
  "${SCRIPT_DIR}/watch_head_role_s_feature_split_gallery.py"
  --config "${CONFIG}"
)
printf -v watcher_command '%q ' "${watcher[@]}"
if ! tmux list-windows -t "${SESSION}" -F '#W' | rg -Fxq gallery; then
  tmux new-window -d -t "${SESSION}" -n gallery \
    "${watcher_command}; exec bash"
fi

tmux select-window -t "${SESSION}:gallery"
tmux list-windows -t "${SESSION}"
