#!/usr/bin/env bash
set -euo pipefail

repo="zai-org/CogVideoX-2b"
revision="1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01"
output_dir="${1:-/data/gaoya/agent-data/weights/CogVideoX-2b}"

files=(
  model_index.json
  scheduler/scheduler_config.json
  text_encoder/config.json
  text_encoder/model-00001-of-00002.safetensors
  text_encoder/model-00002-of-00002.safetensors
  text_encoder/model.safetensors.index.json
  tokenizer/added_tokens.json
  tokenizer/special_tokens_map.json
  tokenizer/spiece.model
  tokenizer/tokenizer_config.json
  transformer/config.json
  transformer/diffusion_pytorch_model.safetensors
  vae/config.json
  vae/diffusion_pytorch_model.safetensors
)

for relative_path in "${files[@]}"; do
  destination="${output_dir}/${relative_path}"
  mkdir -p "$(dirname "${destination}")"
  echo "Downloading ${relative_path}"
  curl --location --fail --retry 8 --retry-delay 5 --continue-at - \
    "https://huggingface.co/${repo}/resolve/${revision}/${relative_path}" \
    --output "${destination}"
done

echo "CogVideoX-2b snapshot saved to ${output_dir}"
