#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_score_extreme_head_ablation_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION=wan_s_score_extreme_ablation_seed851
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_public_head_ablation_case025
GALLERY_CONFIG="${SCRIPT_DIR}/common22_public_head_ablation_case025_with_extremes.json"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/score_extreme_logs"

jobs=(
  "wan_lora top 0"
  "wan_lora bottom 1"
  "xssc top 2"
  "xssc bottom 3"
  "physrvg top 4"
  "physrvg bottom 5"
)
for index in "${!jobs[@]}"; do
  read -r model group gpu <<<"${jobs[$index]}"
  name="${model}_${group}"
  command="cd '${SCRIPT_DIR}' && MODEL='${model}' GROUP='${group}' GPU='${gpu}' bash '${SCRIPT_DIR}/run_score_extreme_head_ablation_job.sh' 2>&1 | tee '${OUTPUT_ROOT}/score_extreme_logs/${name}.log'"
  if (( index == 0 )); then
    tmux new-session -d -s "${SESSION}" -n "${name}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${name}" "${command}"
  fi
done
tmux new-window -t "${SESSION}" -n gallery \
  "cd '${SCRIPT_DIR}' && CONFIG='${GALLERY_CONFIG}' INTERVAL=10 bash '${SCRIPT_DIR}/watch_common22_public_head_gallery.sh'"
echo "session=${SESSION}"
echo "gallery=http://127.0.0.1:8944/multiseed/"
