#!/usr/bin/env bash
# Stage1 continuation on phys_state_episode with 704x1280.
# Uses the object-centric trainer and initializes Wan LoRA from the provided
# step-010000 checkpoint.
set -euo pipefail

ACCELERATE_BIN=/data/gaoya/miniconda3/envs/wan/bin/accelerate
TRAIN_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_context_video_wan.py
CONFIG=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_stage1_adapters_gpu67.yaml
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints
CHECKPOINT_SUBDIR=stage1

CUDA_VISIBLE_DEVICES=6,7 "${ACCELERATE_BIN}" launch --multi_gpu --num_processes 2 --num_machines 1 "${TRAIN_SCRIPT}" \
  --config "${CONFIG}"
