#!/usr/bin/env bash
# Continue Wan2.2 TI2V LoRA training on the raw phys-state simulation videos.
# Starts from the OpenVid-mixed LoRA checkpoint and keeps the train0419_reference
# training stack unchanged apart from the dataset source and output directory.
set -euo pipefail

ACCELERATE_BIN=/data/gaoya/miniconda3/envs/wan/bin/accelerate
TRAIN_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/train.py
DATASET_CONFIG=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/dataset_raw_phys_state_config.json
OUTPUT_DIR=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24
INIT_LORA=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors

EXTRA_ARGS=()
if [[ -f "${OUTPUT_DIR}/training_state.pt" || -d "${OUTPUT_DIR}/checkpoints" ]]; then
  EXTRA_ARGS+=(--resume_from "${OUTPUT_DIR}")
else
  EXTRA_ARGS+=(--lora_checkpoint "${INIT_LORA}")
fi

CUDA_VISIBLE_DEVICES=1,2 "${ACCELERATE_BIN}" launch --multi_gpu --num_processes 2 --num_machines 1 "${TRAIN_SCRIPT}" \
  --diffsynth_root /home/gaoya/Code_Video/DiffSynth-Studio-main \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --dataset_base_path "${DATASET_CONFIG}" \
  --dataset_metadata_path "" \
  --height 576 \
  --width 1024 \
  --num_frames 24 \
  --max_train_steps 10000 \
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
  --num_epochs 10 \
  --gradient_accumulation_steps 4 \
  --save_steps 500 \
  --benchmark_every_steps 1000 \
  --benchmark_meta_list_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_fixed24.txt \
  --benchmark_cuda_visible_devices 1,2 \
  --benchmark_context_frames 8 \
  --benchmark_num_frames 24 \
  --benchmark_height 576 \
  --benchmark_width 1024 \
  --benchmark_fps 30 \
  --benchmark_num_inference_steps 50 \
  --benchmark_cfg_scale 5.0 \
  --benchmark_seed 42 \
  --benchmark_script_path /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py \
  --benchmark_output_subdir fixed24_generation \
  --validation_every_steps 2000 \
  --validation_script_path /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/run_validation_vbench.py \
  --validation_meta_list_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_validation100.txt \
  --validation_context_frames_list 0,1,2,4,6,8 \
  --validation_output_subdir validation100_vbench \
  --validation_vbench_config_path /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/vbench_paths.yaml \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path "${OUTPUT_DIR}" \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 \
  --extra_inputs input_image \
  --report_to wandb \
  --wandb_project phys_state_wan22_continue \
  --wandb_mode offline \
  "${EXTRA_ARGS[@]}"
