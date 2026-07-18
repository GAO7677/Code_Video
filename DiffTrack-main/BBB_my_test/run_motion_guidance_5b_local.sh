#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/gaoya/Code_Video/DiffTrack-main"
SCRIPT_PATH="${REPO_DIR}/BBB_my_test/motion_guidance_5b_local.py"
MODEL_DIR="${MODEL_DIR:-/data/gaoya/ckpt/zai-org-CogVideoX-5b}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/outputs/difftrack_motion_guidance_5b}"
PROMPT_PATH="${PROMPT_PATH:-${REPO_DIR}/dataset/cag_prompts.txt}"
CONDA_BIN="/home/gaoya/miniconda3/bin/conda"
CONDA_ENV="${CONDA_ENV:-bagel}"
VISIBLE_GPU="${CUDA_VISIBLE_DEVICES:-4}"

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

while true; do
  missing=()
  for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
      missing+=("${path}")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    break
  fi

  printf 'Waiting for model download to finish. Missing files:\n'
  printf '  %s\n' "${missing[@]}"
  sleep 30
done

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
  --pag_scale 1 \
  --cfg_scale 6 \
  --device cuda:0
