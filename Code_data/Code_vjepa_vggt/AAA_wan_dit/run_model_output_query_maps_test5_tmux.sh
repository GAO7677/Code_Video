#!/usr/bin/env bash
set -euo pipefail

# SESSION=wan_model_query_maps_test5 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_model_output_query_maps_test5_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
INPUT_LIST=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt
SOURCE_ROOT=/data/gaoya/agent-data/outputs/wan_dit_ball_query_attention/test5_allblocks_stability/generated
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_model_specific_query_maps/test5
SESSION="${SESSION:-wan_model_query_maps_test5}"

tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}/logs"

run_command() {
  local model="$1"
  local gpu="$2"
  local video_root="$3"
  printf \
    "CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q %q %q --input-list %q --model %q --video-root %q --output-dir %q 2>&1 | tee %q" \
    "${gpu}" \
    "${PROJECT_ROOT}" \
    "${PYTHON}" \
    "${SCRIPT_DIR}/build_model_output_sam2_query_map.py" \
    "${INPUT_LIST}" \
    "${model}" \
    "${video_root}" \
    "${OUTPUT_ROOT}/${model}" \
    "${OUTPUT_ROOT}/logs/${model}.log"
}

tmux new-session -d -s "${SESSION}" -n wan_lora \
  "$(run_command wan_lora 3 "${SOURCE_ROOT}/wan_lora")"
tmux new-window -t "${SESSION}" -n physrvg \
  "$(run_command physrvg 4 "${SOURCE_ROOT}/physrvg")"
tmux new-window -t "${SESSION}" -n xssc \
  "$(run_command xssc 6 "${SOURCE_ROOT}/xssc")"

echo "started ${SESSION}: wan_lora=GPU3, physrvg=GPU4, xssc=GPU6"
echo "logs: ${OUTPUT_ROOT}/logs"
