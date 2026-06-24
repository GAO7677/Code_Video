#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=6,7 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate launch \
  --multi_gpu --num_processes 2 --gpu_ids 6,7 --mixed_precision bf16 \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_context_video_wan.py \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml
