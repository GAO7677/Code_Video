#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_common_stc_all_heads_qk_seed851_tmux.sh
# tmux attach -t wan_common_stc_qk_851

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SESSION="${SESSION:-wan_common_stc_qk_851}"
ROOT=/data/gaoya/agent-data/outputs/wan_dit_common_stc_all_heads_qk_seed851
GALLERY=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/common-stc-all-heads-qk-seed851
REPORT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/partial_analysis/snapshot_20260728T0245Z/common22/aggregate_heads.csv
INPUT_LIST="${SCRIPT_DIR}/common22_public_head_ablation_case025.txt"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "${ROOT}/logs" "${ROOT}/state" "${ROOT}/heatmaps"
"${PYTHON}" "${SCRIPT_DIR}/build_common_stc_all_heads_selection.py" \
  --report "${REPORT}" --input-list "${INPUT_LIST}" \
  --output-dir "${ROOT}" --seed 851
rm -f "${ROOT}/state/"*.complete "${ROOT}/state/"*.failed

tmux new-session -d -s "${SESSION}" -n wan_lora \
  "MODEL=wan_lora GPU=3 bash '${SCRIPT_DIR}/run_common_stc_all_heads_qk_model_worker.sh'; exec bash"
tmux new-window -t "${SESSION}" -n xssc \
  "MODEL=xssc GPU=6 bash '${SCRIPT_DIR}/run_common_stc_all_heads_qk_model_worker.sh'; exec bash"
tmux new-window -t "${SESSION}" -n physrvg \
  "MODEL=physrvg GPU=7 bash '${SCRIPT_DIR}/run_common_stc_all_heads_qk_model_worker.sh'; exec bash"
tmux new-window -t "${SESSION}" -n render \
  "bash '${SCRIPT_DIR}/wait_render_common_stc_all_heads_qk.sh'; exec bash"

echo "tmux session: ${SESSION}"
echo "Wan+LoRA: GPU3; Wan+xSSC: GPU6; PhysRVG: GPU7; GPU4 unused"
echo "gallery: http://127.0.0.1:8946/common-stc-all-heads-qk-seed851/"
