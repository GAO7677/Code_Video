#!/usr/bin/env bash



export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/gaoya/Code_Video/DreamWorld-main:${PYTHONPATH:-}
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES=0

cd /home/gaoya/Code_Video/DreamWorld-main

/home/gaoya/miniconda3/envs/wan-cu128/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:19242 \
  /home/gaoya/Code_Video/DreamWorld-main/script/inference/inference.py \
  --parallel_backend ptd \
  --pp_degree 1 \
  --dp_degree 1 \
  --dp_shards 1 \
  --cp_degree 1 \
  --tp_degree 1 \
  --model_name wan \
  --pretrained_model_name_or_path /data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers \
  --enable_slicing \
  --enable_tiling \
  --dino_in_channels 8 \
  --dino_out_channels 8 \
  --vggt_in_channels 8 \
  --vggt_out_channels 8 \
  --flow_in_channels 16 \
  --flow_out_channels 16 \
  --inference_type text_to_video \
  --dataset_file /home/gaoya/Code_Video/DreamWorld-main/script/inference/inference_demo_min.json \
  --attn_provider native \
  --allow_tf32 \
  --float32_matmul_precision high \
  --seed 42 \
  --tracker_name dreamworld-inference \
  --lora_path /data/gaoya/ckpt/TeanABU-DreamWorld/pytorch_lora_weights.safetensors \
  --output_dir /home/gaoya/Code_Video/DreamWorld-main/outputs/demo_min \
  --init_timeout 600 \
  --nccl_timeout 600 \
  --report_to none
