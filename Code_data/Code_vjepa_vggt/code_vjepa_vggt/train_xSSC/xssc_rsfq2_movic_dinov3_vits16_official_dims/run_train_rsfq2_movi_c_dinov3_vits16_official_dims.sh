#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_movic_dinov3_vits16_official_dims"
PYTHON_ENV="/home/gaoya/miniconda3/envs/wan-cu128"
SAVE_DIR="/data/gaoya/agent-data/checkpoints/xssc_dinov3_vits16_official_dims"
CONFIG="upstream/config-randsfq/rsfq2_c-movi_c-dinov3_vits16_256-official_dims.py"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DINOV3_CHECKPOINT="/data/gaoya/ckpt/facebook-dinov3-vits16-pretrain-lvd1689m/model.safetensors"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_MODE=online

cd "$ROOT"
exec "$PYTHON_ENV/bin/torchrun" \
    --standalone \
    --nproc_per_node=4 \
    train_ddp_ytvis_hq.py \
    --project xssc_dinov3 \
    --seed 42 \
    --cfg-file "$CONFIG" \
    --data-dir /data/gaoya/dataset \
    --save-dir "$SAVE_DIR"
