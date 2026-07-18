#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/gaoya/Code_Video/DiffTrack-main"
SCRIPT_PATH="${REPO_DIR}/BBB_my_test/motion_guidance_5b_local.py"
MODEL_DIR="${MODEL_DIR:-/data/gaoya/ckpt/zai-org-CogVideoX-5b}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/dataset/cag_prompts.txt}"
PAG_OUTPUT_DIR="${PAG_OUTPUT_DIR:-/data/gaoya/agent-data/outputs/difftrack_motion_guidance_5b}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/outputs/difftrack_motion_guidance_5b_baseline}"
CONDA_BIN="/home/gaoya/miniconda3/bin/conda"
CONDA_ENV="${CONDA_ENV:-bagel}"
VISIBLE_GPU="${CUDA_VISIBLE_DEVICES:-1}"

required_files=(
  "${MODEL_DIR}/model_index.json"
  "${MODEL_DIR}/scheduler/scheduler_config.json"
  "${MODEL_DIR}/tokenizer/spiece.model"
  "${MODEL_DIR}/vae/diffusion_pytorch_model.safetensors"
  "${MODEL_DIR}/text_encoder/model-00001-of-00002.safetensors"
  "${MODEL_DIR}/text_encoder/model-00002-of-00002.safetensors"
  "${MODEL_DIR}/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
  "${MODEL_DIR}/transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing model file: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${PAG_OUTPUT_DIR}" ]]; then
  echo "PAG output directory not found: ${PAG_OUTPUT_DIR}" >&2
  exit 1
fi

case_count=$(find "${PAG_OUTPUT_DIR}" -maxdepth 1 -name 'video_*.mp4' | wc -l)
if [[ "${case_count}" -eq 0 ]]; then
  echo "No PAG videos found under ${PAG_OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/diffusers/src:${REPO_DIR}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"

exec "${CONDA_BIN}" run -n "${CONDA_ENV}" python "${SCRIPT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_version 5b \
  --model_path "${MODEL_DIR}" \
  --txt_path "${PROMPT_PATH}" \
  --pag_layers 13 17 21 \
  --pag_scale 0 \
  --cfg_scale 6 \
  --device cuda:0 \
  --max_prompts "${case_count}"
