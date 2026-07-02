#!/usr/bin/env bash
set -euo pipefail

ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
TRAIN_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_stage1b_context_only_no_gt_box_diffsynth.py
OUTPUT_DIR=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1b_context_only_no_gt_box_diffsynth
STAGE1A_CKPT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
LATEST_STATE=""
if [ -d "${OUTPUT_DIR}/checkpoints" ]; then
    LATEST_STATE=$(find "${OUTPUT_DIR}/checkpoints" -path '*/training_state.pt' -type f | sort -V | tail -n 1 || true)
fi
if [ -n "${LATEST_STATE}" ]; then
    echo "[resume] found state: ${LATEST_STATE}"
    RESUME_ARGS=(--stage2_resume_from "${LATEST_STATE}")
else
    echo "[fresh] no DiffSynth resume state found"
fi

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2,3,6,7 "${ACCELERATE_BIN}" launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 "${TRAIN_SCRIPT}" \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --phys_state_root /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500 \
  --phys_state_split train \
  --height 512 \
  --width 896 \
  --fixed_num_context_frames 8 \
  --batch_size 1 \
  --dataset_num_workers 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --num_epochs 100 \
  --max_train_steps 20000 \
  --gradient_accumulation_steps 1 \
  --optimizer_type paged_adamw8bit \
  --max_grad_norm 1.0 \
  --save_steps 500 \
  --max_checkpoints_keep 10 \
  --output_path "${OUTPUT_DIR}" \
  --head_resume_from "${STAGE1A_CKPT}" \
  --lora_rank 32 \
  --lora_alpha 32 \
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
  --cond_proj_dim 4096 \
  --lambda_main 1.0 \
  --lambda_track_aux 0.0 \
  --lambda_box_aux 0.0 \
  --lambda_depth_aux 0.0 \
  --report_to wandb \
  --wandb_project vjepa_vggt_wan \
  --wandb_name pybullet0629_teacher_student_stage1b_context_only_no_gt_box_diffsynth_gpu2367 \
  --wandb_mode online \
  "${RESUME_ARGS[@]}" \
  "$@"
