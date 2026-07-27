#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash run_consistent_category_head_zero_case001460_one.sh CATEGORY GPU_ID
# CATEGORY: S | ST | T | P | C | G

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 CATEGORY GPU_ID" >&2
  exit 2
fi

CATEGORY="${1^^}"
GPU_ID="$2"
case "${CATEGORY}" in S|ST|T|P|C|G) ;; *) exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

INPUT_LIST="${SCRIPT_DIR}/ball_query_case001460.txt"
CLASSIFICATION_METADATA=/data/gaoya/agent-data/outputs/wan_dit_allblock_head_roles/case001460_latent_aligned_wan_lora/metadata.json
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/agent-data/outputs/wan_dit_consistent_category_head_ablation/case001460_wan_lora}"
TAG="self_attn_consistent_head_zero_category_${CATEGORY,,}"
OUTPUT_ROOT="${OUTPUT_BASE}/${TAG}"
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
WAN_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500
NEGATIVE_PROMPT="模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"

declare -A EXPECTED_COUNTS=(
  [S]=110
  [ST]=6
  [T]=40
  [P]=41
  [C]=21
  [G]=120
)

mkdir -p "${OUTPUT_ROOT}"
test -s "${INPUT_LIST}"
test -s "${CLASSIFICATION_METADATA}"
{
  echo "model=wan_lora"
  echo "category=${CATEGORY}"
  echo "num_targets=${EXPECTED_COUNTS[${CATEGORY}]}"
  echo "gpu=${GPU_ID}"
  echo "input_list=${INPUT_LIST}"
  echo "classification_metadata=${CLASSIFICATION_METADATA}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "height=512"
  echo "width=896"
  echo "num_frames=49"
  echo "context_frames=8"
  echo "conditioning_mode=context_aware"
  echo "context_resize_mode=crop"
  echo "num_inference_steps=40"
  echo "cfg_scale=5.0"
  echo "seed=42"
  echo "negative_prompt=${NEGATIVE_PROMPT}"
} > "${OUTPUT_ROOT}/ablation_config.txt"

exec env \
  PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON}" "${SCRIPT_DIR}/infer_wan_lora_consistent_head_ablation.py" \
  --head-category "${CATEGORY}" \
  --classification-metadata "${CLASSIFICATION_METADATA}" \
  --expected-target-count "${EXPECTED_COUNTS[${CATEGORY}]}" \
  --weights-root "${WAN_LORA_ROOT}" \
  --input-json-list-path "${INPUT_LIST}" \
  --model-name "wan_lora_${TAG}" \
  --wan-root "${WAN_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --runtime-root "${OUTPUT_ROOT}/_runtime" \
  --device cuda --height 512 --width 896 --num-frames 49 \
  --context-frames 8 --conditioning-mode context_aware \
  --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
  --fps 30 --seed 42 --negative-prompt "${NEGATIVE_PROMPT}"
