#!/usr/bin/env bash
# 该脚本用于启动 train.py 训练；当前配置为 4x4090、每卡 batch=1、grad_accum=4，因此 effective batch=16。
# 输入为 OpenVid + MOVI-D + Genesis rigid 混合数据集配置和 /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B 模型权重，输出为 /data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora 下的 checkpoint、固定可视化样例和 validation+VBench 结果。
# sh /home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_train.sh
set -euo pipefail

ACCELERATE_BIN=/data/gaoya/miniconda3/envs/wan/bin/accelerate
TRAIN_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419/train.py
OUTPUT_DIR=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora

EXTRA_ARGS=()
if [[ -f "${OUTPUT_DIR}/training_state.pt" || -d "${OUTPUT_DIR}/checkpoints" ]]; then
  EXTRA_ARGS+=(--resume_from "${OUTPUT_DIR}")
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 "${ACCELERATE_BIN}" launch --multi_gpu --num_processes 4 --num_machines 1 "${TRAIN_SCRIPT}" \
  --diffsynth_root /home/gaoya/Code_Video/DiffSynth-Studio-main \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --dataset_base_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/dataset_mix_config.json \
  --dataset_metadata_path "" \
  --height 384 \
  --width 672 \
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
  --save_steps 1000 \
  --benchmark_every_steps 1000 \
  --benchmark_meta_list_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_fixed24.txt \
  --benchmark_cuda_visible_devices 0,1,2,3 \
  --benchmark_context_frames 8 \
  --benchmark_num_frames 24 \
  --benchmark_height 384 \
  --benchmark_width 672 \
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
  --output_path "${OUTPUT_DIR}" \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 \
  --extra_inputs input_image \
  --report_to wandb \
  --wandb_project openvid-movid-genesis-wan22 \
  --wandb_mode offline \
  "${EXTRA_ARGS[@]}"
