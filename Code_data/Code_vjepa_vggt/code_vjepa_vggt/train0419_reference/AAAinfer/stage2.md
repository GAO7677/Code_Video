CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_stage2_adapters.py \
  --stage2-checkpoint /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/pybullet0613_stage2_adapters_gpu67/step_0000600.pt \
  --stage1-checkpoint /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-001000/checkpoint.safetensors \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_stage2_adapters_gpu67.yaml \
  --prompt "A sphere rolls after landing on the platform and leaves the support surface, testing support switching." \
  --context-video /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_000335/context_video.mp4 \
  --output-dir /data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/stage2_step600_proper_infer \
  --num-frames 24 \
  --sampling-mode prefix \
  --save-raw