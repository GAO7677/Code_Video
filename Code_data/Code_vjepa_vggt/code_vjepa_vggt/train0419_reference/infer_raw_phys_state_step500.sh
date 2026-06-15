
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_train/train_0419:/home/gaoya/Code_Video/DiffSynth-Studio-main \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --lora_path /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
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
  --device cuda \
  --sample_id bus \
  --context_path /data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output/GT/Dynamic_Tracking/bus_0s_1p5s.mp4 \
  --prompt "bus" \
  --output_video_path /data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/500_bus.mp4 \




  --context_path /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_000335/context_video.mp4 \
  --prompt "A sphere rolls after landing on the platform and leaves the support surface, testing support switching." \

  --prompt "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement.", \
  --context_path /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/AAAsource/physicIQ_0002_clip_2p5s_3p5s.mp4 \










CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_train/train_0419:/home/gaoya/Code_Video/DiffSynth-Studio-main \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --lora_path /data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors \
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
  --model_name openvid-step-010000 \
  --device cuda \
  --sample_id bus \
  --context_path /data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output/GT/Dynamic_Tracking/bus_0s_1p5s.mp4 \
  --prompt "bus" \
  --output_video_path /data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/10000_bus.mp4 \
