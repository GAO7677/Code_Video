#!/usr/bin/env bash
set -euo pipefail

# SESSION=wan_anchor_t2_allblock_test5 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_anchor_t2_allblock_test5_gpu4_serial_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-wan_anchor_t2_allblock_test5}"
GPU="${GPU:-4}"
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_anchor_t2_attention/test5_allblocks
QUERY_ROOT=/data/gaoya/agent-data/outputs/wan_dit_model_specific_query_maps/test5
INPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_anchor_t2_attention/input_lists
RUNNER="${SCRIPT_DIR}/run_allblock_ball_query_test5.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"

command_text="set -euo pipefail;"
for model in wan_lora xssc physrvg; do
  query_map="${QUERY_ROOT}/${model}/query_map.json"
  if [[ "${model}" == "wan_lora" ]]; then
    input_list="${INPUT_ROOT}/wan_lora_test5_anchor_t2_19.txt"
  else
    input_list="${INPUT_ROOT}/test5_anchor_t2_20.txt"
  fi
  test -s "${query_map}"
  test -s "${input_list}"
  command_text+=" echo '[anchor-t2-attention] start ${model}';"
  command_text+=" MODEL=${model} GPU=${GPU} INPUT_LIST=${input_list} QUERY_MAP=${query_map} QUERY_MODE=anchor_t2 OUTPUT_ROOT=${OUTPUT_ROOT} bash ${RUNNER}"
  command_text+=" 2>&1 | tee ${OUTPUT_ROOT}/logs/${model}.log;"
  command_text+=" echo '[anchor-t2-attention] complete ${model}';"
done

tmux new-session -d -s "${SESSION}" -n capture "${command_text}"

echo "started ${SESSION} on GPU${GPU}: wan_lora=19, xssc=20, physrvg=20"
echo "output: ${OUTPUT_ROOT}"
echo "attach: tmux attach -t ${SESSION}"
