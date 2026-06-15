# sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/infer.sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_train/train_0419:/home/gaoya/Code_Video/DiffSynth-Studio-main \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --lora_path /data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors \
  --context_path /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_000335/context_video.mp4 \
  --output_video_path /data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/train0419_step10000_sample000335.mp4 \
  --prompt "A sphere rolls after landing on the platform and leaves the support surface, testing support switching." \
  --sample_id sample_000335 \
  --dataset_name physv_0613pybullet \
  --height 704 \
  --width 1280 \
  --num_frames 40 \
  --context_frames 7 \
  --fps 30 \
  --num_inference_steps 50 \
  --cfg_scale 5.0 \
  --seed 42 \
  --conditioning_mode context_aware \
  --model_name step-010000 \
  --device cuda