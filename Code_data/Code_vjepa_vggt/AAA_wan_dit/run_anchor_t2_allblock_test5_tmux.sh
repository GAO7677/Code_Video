#!/usr/bin/env bash
set -euo pipefail

# SESSION=wan_anchor_t2_allblock_test5 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_anchor_t2_allblock_test5_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-wan_anchor_t2_allblock_test5}"
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_anchor_t2_attention/test5_allblocks
QUERY_ROOT=/data/gaoya/agent-data/outputs/wan_dit_model_specific_query_maps/test5
INPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_anchor_t2_attention/input_lists
RUNNER="${SCRIPT_DIR}/run_allblock_ball_query_test5.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"

run_command() {
  local model="$1"
  local gpu="$2"
  local input_list="$3"
  local query_map="${QUERY_ROOT}/${model}/query_map.json"
  test -s "${input_list}"
  test -s "${query_map}"
  printf \
    "MODEL=%q GPU=%q INPUT_LIST=%q QUERY_MAP=%q QUERY_MODE=anchor_t2 OUTPUT_ROOT=%q bash %q 2>&1 | tee %q" \
    "${model}" \
    "${gpu}" \
    "${input_list}" \
    "${query_map}" \
    "${OUTPUT_ROOT}" \
    "${RUNNER}" \
    "${OUTPUT_ROOT}/logs/${model}.log"
}

tmux new-session -d -s "${SESSION}" -n wan_lora \
  "$(run_command wan_lora 4 "${INPUT_ROOT}/wan_lora_test5_anchor_t2_19.txt")"
tmux new-window -t "${SESSION}" -n xssc \
  "$(run_command xssc 5 "${INPUT_ROOT}/test5_anchor_t2_20.txt")"
tmux new-window -t "${SESSION}" -n physrvg \
  "$(run_command physrvg 6 "${INPUT_ROOT}/test5_anchor_t2_20.txt")"

echo "started ${SESSION}: wan_lora=GPU4 (19), xssc=GPU5 (20), physrvg=GPU6 (20)"
echo "output: ${OUTPUT_ROOT}"
echo "attach: tmux attach -t ${SESSION}"
