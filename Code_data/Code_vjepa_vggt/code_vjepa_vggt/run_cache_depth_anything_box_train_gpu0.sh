#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/cache_depth_anything_box_from_npz.py

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${SCRIPT}" \
  --dataset-root /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500 \
  --split train \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/depth_anything_cache \
  --gpu 0 \
  --num-shards "${NUM_SHARDS:-4}" \
  --shard-index "${SHARD_INDEX:-0}"
