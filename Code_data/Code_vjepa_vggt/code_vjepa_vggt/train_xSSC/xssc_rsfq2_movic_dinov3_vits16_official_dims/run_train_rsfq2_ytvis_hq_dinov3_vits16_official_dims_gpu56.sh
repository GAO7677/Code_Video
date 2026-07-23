#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_movic_dinov3_vits16_official_dims"
PYTHON_ENV="/home/gaoya/miniconda3/envs/wan-cu128"
CONFIG="upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vits16_256-official_dims.py"
DATA_DIR="${DATA_DIR:-/data/gaoya/dataset}"
SAVE_DIR="${SAVE_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/restart_save1000_20260720T140029Z/ytvis_hq_dinov3_vits16_official_dims_b192_acc1_20260723T125549Z}"
MAX_STEP="${MAX_STEP:-15000}"
WANDB_PROJECT="${WANDB_PROJECT:-xssc_dinov3}"
WANDB_MODE="${WANDB_MODE:-online}"

export CUDA_VISIBLE_DEVICES=5,6
export DINOV3_CHECKPOINT="/data/gaoya/ckpt/facebook-dinov3-vits16-pretrain-lvd1689m/model.safetensors"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export WANDB_MODE

cd "$ROOT"
exec "$PYTHON_ENV/bin/torchrun" \
    --standalone \
    --nproc_per_node=2 \
    train_ddp_ytvis_hq.py \
    --project "$WANDB_PROJECT" \
    --seed 42 \
    --cfg-file "$CONFIG" \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --max-step "$MAX_STEP"
