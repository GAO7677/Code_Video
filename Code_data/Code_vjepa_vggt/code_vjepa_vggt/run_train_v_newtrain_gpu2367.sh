#!/usr/bin/env bash
set -euo pipefail
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_gpu2367.sh
ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
TRAIN_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_v_newtrain.py
OUTPUT_DIR=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu2367

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2,3,6,7 "${ACCELERATE_BIN}" launch --multi_gpu --num_processes 4 --num_machines 1 "${TRAIN_SCRIPT}" \
  --diffsynth_root /home/gaoya/Code_Video/DiffSynth-Studio-main \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --dataset_type phys_state_episode \
  --phys_state_root /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500 \
  --phys_state_split train \
  --height 512 \
  --width 896 \
  --num_frames 24 \
  --fixed_num_context_frames 8 \
  --max_train_steps 20000 \
  --dataset_num_workers 0 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --num_epochs 100 \
  --gradient_accumulation_steps 1 \
  --find_unused_parameters \
  --save_steps 1000 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path "${OUTPUT_DIR}" \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 \
  --extra_inputs input_image \
  --enable_object_branch \
  --object_num_queries 8 \
  --aux_max_objects 4 \
  --jepa_ckpt_path /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth \
  --jepa_input_size 384 \
  --jepa_patch_size 16 \
  --jepa_tubelet_size 2 \
  --cotracker_checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth \
  --cotracker_input_h 384 \
  --cotracker_input_w 512 \
  --cotracker_window_len 60 \
  --object_pooler_latent_dim 16 \
  --cond_proj_dim 4096 \
  --jepa_window_radius 1 \
  --latent_window_radius 1 \
  --lambda_track_aux 0.1 \
  --lambda_box_aux 0.1 \
  --lambda_depth_aux 0.0 \
  --report_to wandb \
  --wandb_project vjepa_vggt_wan \
  --wandb_name pybullet0624_diffsynth_object_v_newtrain_gpu2367 \
  --wandb_mode online
