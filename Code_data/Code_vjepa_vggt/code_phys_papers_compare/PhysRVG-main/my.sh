  source /home/gaoya/miniconda3/etc/profile.d/conda.sh
  conda activate vjepa2
  PYTHONNOUSERSITE=1 python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main/inference.py \
    --device 0 \
    --model_id /data/gaoya/ckpt/HappyP4nda-PhysRVG/Wan2.2-TI2V-5B-Diffusers \
    --dit_checkpoint /data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors \
    --lora_checkpoint /data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint \
    --video_path /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_001460/source_video_first8.mp4 \
    --video_save_dir /data/gaoya/agent-data/outputs/physrvg_output