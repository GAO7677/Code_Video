#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256"
PYTHON_ENV="/home/gaoya/miniconda3/envs/wan-cu128"
CHECKPOINT="/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/restart_save1000_20260720T140029Z/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512/42/step-015000.pth"
SAVE_DIR="/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/restart_save1000_20260720T140029Z/movi_c_transfer15000_b64_acc3_20260721T134713Z"
CONFIG="upstream/config-randsfq/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"

export CUDA_VISIBLE_DEVICES=5,6
export DINOV3_CHECKPOINT="/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_MODE=online

cd "$ROOT"
exec "$PYTHON_ENV/bin/torchrun" \
    --standalone \
    --nproc_per_node=2 \
    train_ddp_ytvis_hq.py \
    --project xssc_dinov3 \
    --seed 42 \
    --cfg-file "$CONFIG" \
    --data-dir /data/gaoya/dataset \
    --save-dir "$SAVE_DIR" \
    --ckpt-file "$CHECKPOINT"
