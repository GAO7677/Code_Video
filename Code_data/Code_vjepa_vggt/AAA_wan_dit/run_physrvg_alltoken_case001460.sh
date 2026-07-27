#!/usr/bin/env bash
set -euo pipefail

# GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physrvg_alltoken_case001460.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
PHYSRVG_ROOT="${PROJECT_ROOT}/code_phys_papers_compare/PhysRVG-main"
PYTHON=/data/gaoya/miniconda3/envs/vjepa2/bin/python

GPU="${GPU:-3}"
INPUT_LIST="${SCRIPT_DIR}/ball_query_case001460.txt"
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/physrvg_alltoken_attention/case001460
BLOCKS="$(seq -s, 0 29)"

MODEL_ID=/data/gaoya/ckpt/HappyP4nda-PhysRVG/Wan2.2-TI2V-5B-Diffusers
DIT_CHECKPOINT=/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors
LORA_CHECKPOINT=/data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint

mkdir -p "${OUTPUT_ROOT}/logs"

env \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONPATH="${PHYSRVG_ROOT}:${SCRIPT_DIR}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" "${SCRIPT_DIR}/capture_physrvg_alltoken_attention.py" \
  --attention-output-root "${OUTPUT_ROOT}/attention" \
  --attention-blocks "${BLOCKS}" \
  --attention-step 25 \
  --physrvg-root "${PHYSRVG_ROOT}" \
  --input-json-list-paths "${INPUT_LIST}" \
  --output-root "${OUTPUT_ROOT}/generated" \
  --model-id "${MODEL_ID}" \
  --dit-checkpoint "${DIT_CHECKPOINT}" \
  --lora-checkpoint "${LORA_CHECKPOINT}" \
  --device cuda:0 \
  --height 512 --width 896 --num-frames 49 --fps 30 \
  --num-inference-steps 40 --guidance-scale 5.0 --seed 42 --force \
  2>&1 | tee "${OUTPUT_ROOT}/logs/case001460.log"
