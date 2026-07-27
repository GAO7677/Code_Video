#!/usr/bin/env bash
set -euo pipefail

# GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_two_ball_wan_lora_case001460.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

GPU="${GPU:-3}"
CASE=0613pybullet_sample_001460_w002
INPUT_LIST="${SCRIPT_DIR}/ball_query_case001460.txt"
QUERY_MAP=/data/gaoya/agent-data/outputs/wan_dit_two_ball_query_map/case001460/query_map.json
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_two_ball_attention/case001460
BLOCKS="$(seq -s, 0 29)"
HEADS="$(seq -s, 0 23)"

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
WAN_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500
NEGATIVE_PROMPT="模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"

mkdir -p "${OUTPUT_ROOT}/logs"
test -s "${QUERY_MAP}"

env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}:${SCRIPT_DIR}" \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" "${SCRIPT_DIR}/capture_two_ball_wan_lora_attention.py" \
  --attention-output-root "${OUTPUT_ROOT}/attention" \
  --attention-blocks "${BLOCKS}" \
  --attention-step 25 \
  --attention-query-map "${QUERY_MAP}" \
  --attention-map-heads "${HEADS}" \
  --weights-root "${WAN_LORA_ROOT}" \
  --input-json-list-path "${INPUT_LIST}" \
  --model-name wan_lora_two_ball_attention \
  --wan-root "${WAN_ROOT}" \
  --output-root "${OUTPUT_ROOT}/generated" \
  --runtime-root "${OUTPUT_ROOT}/generated/_runtime" \
  --device cuda --height 512 --width 896 --num-frames 49 \
  --context-frames 8 --conditioning-mode context_aware \
  --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
  --fps 30 --seed 42 --negative-prompt "${NEGATIVE_PROMPT}" --overwrite \
  2>&1 | tee "${OUTPUT_ROOT}/logs/${CASE}.log"
