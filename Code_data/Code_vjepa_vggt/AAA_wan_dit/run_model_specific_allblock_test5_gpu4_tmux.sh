#!/usr/bin/env bash
set -euo pipefail

# SESSION=wan_model_specific_allblock_test5 GPU=4 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_model_specific_allblock_test5_gpu4_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-wan_model_specific_allblock_test5}"
GPU="${GPU:-4}"
INPUT_LIST=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt
QUERY_ROOT=/data/gaoya/agent-data/outputs/wan_dit_model_specific_query_maps/test5
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_model_specific_attention/test5_allblocks
RUNNER="${SCRIPT_DIR}/run_allblock_ball_query_test5.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"

command_text="set -euo pipefail;"
for model in wan_lora xssc physrvg; do
  query_map="${QUERY_ROOT}/${model}/query_map.json"
  test -s "${query_map}"
  command_text+=" echo '[model-specific-attention] start ${model}';"
  command_text+=" MODEL=${model} GPU=${GPU} INPUT_LIST=${INPUT_LIST} QUERY_MAP=${query_map} OUTPUT_ROOT=${OUTPUT_ROOT} bash ${RUNNER}"
  command_text+=" 2>&1 | tee ${OUTPUT_ROOT}/logs/${model}.log;"
  command_text+=" echo '[model-specific-attention] complete ${model}';"
done

tmux new-session -d -s "${SESSION}" -n capture "${command_text}"

echo "started tmux session ${SESSION} on GPU${GPU}"
echo "output: ${OUTPUT_ROOT}"
echo "attach: tmux attach -t ${SESSION}"
