#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_allblock_ball_query_test5_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-wan_test5_allblock_head_stability}"
ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_ball_query_attention/test5_allblocks_stability}"
QUERY_ROOT="${QUERY_ROOT:-${ROOT}/query_map}"

tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
}
mkdir -p "${ROOT}/logs"
tmux new-session -d -s "${SESSION}" -n wan_lora \
  "set -o pipefail; MODEL=wan_lora GPU=3 OUTPUT_ROOT='${ROOT}' QUERY_ROOT='${QUERY_ROOT}' bash '${SCRIPT_DIR}/run_allblock_ball_query_test5.sh' 2>&1 | tee '${ROOT}/logs/wan_lora.log' && MODEL=physrvg GPU=3 OUTPUT_ROOT='${ROOT}' QUERY_ROOT='${QUERY_ROOT}' bash '${SCRIPT_DIR}/run_allblock_ball_query_test5.sh' 2>&1 | tee '${ROOT}/logs/physrvg.log'"
tmux new-window -t "${SESSION}" -n xssc \
  "set -o pipefail; MODEL=xssc GPU=4 OUTPUT_ROOT='${ROOT}' QUERY_ROOT='${QUERY_ROOT}' bash '${SCRIPT_DIR}/run_allblock_ball_query_test5.sh' 2>&1 | tee '${ROOT}/logs/xssc.log'"
echo "started ${SESSION}: Wan+LoRA then PhysRVG on GPU3; Wan+xSSC on GPU4"
