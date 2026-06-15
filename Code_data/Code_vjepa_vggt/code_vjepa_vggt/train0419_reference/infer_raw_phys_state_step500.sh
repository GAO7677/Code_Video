#!/usr/bin/env bash
# Inference for the raw-phys-state Wan LoRA continuation checkpoint at step-000500.
# This keeps the train0419_reference inference stack, but aligns the LoRA path and
# generation size/length with the raw phys-state training run.
# sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/infer_raw_phys_state_step500.sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_train/train_0419:/home/gaoya/Code_Video/DiffSynth-Studio-main \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --lora_path /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
  --context_path /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_000335/context_video.mp4 \
  --output_video_path /data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/raw_phys_state_step500_sample000335.mp4 \
  --prompt "A sphere rolls after landing on the platform and leaves the support surface, testing support switching." \
  --sample_id sample_000335 \
  --dataset_name physv_0613pybullet_raw \
  --height 576 \
  --width 1024 \
  --num_frames 24 \
  --context_frames 8 \
  --fps 30 \
  --num_inference_steps 50 \
  --cfg_scale 5.0 \
  --seed 42 \
  --conditioning_mode context_aware \
  --model_name step-000500 \
  --device cuda
