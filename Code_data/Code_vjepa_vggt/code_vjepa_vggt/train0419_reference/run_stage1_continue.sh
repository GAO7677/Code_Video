#!/usr/bin/env bash
# Continue Wan LoRA stage1 training from the existing step-010000 checkpoint.
# Output checkpoints will be written under:
#   /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/stage1
# If you want to resume from a later stage1 checkpoint instead, change RESUME_FROM
# to the new stage1 output directory.
set -euo pipefail

ACCELERATE_BIN=/data/gaoya/miniconda3/envs/wan/bin/accelerate
TRAIN_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/train.py
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints
CHECKPOINT_SUBDIR=stage1
RESUME_FROM=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000

CUDA_VISIBLE_DEVICES=0,1,2,3 "${ACCELERATE_BIN}" launch --multi_gpu --num_processes 4 --num_machines 1 "${TRAIN_SCRIPT}" \
  --diffsynth_root /home/gaoya/Code_Video/DiffSynth-Studio-main \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --dataset_base_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/dataset_mix_config.json \
  --dataset_metadata_path "" \
  --height 704 \
  --width 1280 \
  --num_frames 24 \
  --max_train_steps 20000 \
  --context_sampling_profile mixed_modes \
  --min_context_frames 1 \
  --max_context_ratio 0.5 \
  --context_reference_frames 49 \
  --context_reference_prefixes 1,4,8,12,16 \
  --prefix_context_ratio 0.55 \
  --first_frame_context_ratio 0.20 \
  --sparse_context_ratio 0.15 \
  --random_context_ratio 0.05 \
  --no_context_ratio 0.05 \
  --dataset_repeat 1 \
  --dataset_num_workers 0 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --num_epochs 20 \
  --gradient_accumulation_steps 4 \
  --save_steps 1000 \
  --benchmark_every_steps 1000 \
  --benchmark_meta_list_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_fixed24.txt \
  --benchmark_cuda_visible_devices 0,1,2,3 \
  --benchmark_context_frames 8 \
  --benchmark_num_frames 24 \
  --benchmark_height 704 \
  --benchmark_width 1280 \
  --benchmark_fps 8 \
  --benchmark_num_inference_steps 50 \
  --benchmark_cfg_scale 5.0 \
  --benchmark_seed 42 \
  --benchmark_script_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/batch_eval_lora.py \
  --benchmark_output_subdir fixed24_generation \
  --validation_every_steps 2000 \
  --validation_script_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_validation_vbench.py \
  --validation_meta_list_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_validation100.txt \
  --validation_context_frames_list 0,1,2,4,6,8 \
  --validation_output_subdir validation100_vbench \
  --validation_vbench_config_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/vbench_paths.yaml \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path "${OUTPUT_ROOT}" \
  --checkpoint_output_subdir "${CHECKPOINT_SUBDIR}" \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 \
  --extra_inputs input_image \
  --report_to wandb \
  --wandb_project openvid-movid-genesis-wan22 \
  --wandb_mode offline \
  --resume_from "${RESUME_FROM}"
