#!/usr/bin/env bash
# Stage2 training for raw phys-state Wan LoRA continuation.
# This script only passes arguments supported by train_stage2.py.

set -euo pipefail

CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision bf16 \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/train_stage2.py \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_stage2_adapters_gpu67.yaml \
  --stage1-checkpoint /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-001000/checkpoint.safetensors
