#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/cache_vggt_dense_features.py \
  --dataset-root /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500 \
  --dataset-split train \
  --dataset-start 0 \
  --dataset-end 900 \
  --dataset-limit 900 \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/vggt_cache \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml
