#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6,7
export CODEX_DEBUG_TRAINER_INIT=1
export CODEX_DEBUG_RUNNER_INIT=1
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt

/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate launch \
  --multi_gpu --num_processes 2 --gpu_ids 6,7 --mixed_precision bf16 \
  --main_process_port 29525 \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_context_video_wan.py \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml \
  --resume-checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000940.pt
