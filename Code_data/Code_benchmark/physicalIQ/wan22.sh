CUDA_VISIBLE_DEVICES=5,6,7 python /home/gaoya/Code_Video/Code_data/Code_benchmark/physicalIQ/wan22_ti2v_physics_iq_eval_multigpu.py \
  --multi_gpu \
  --model_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --output_root /data/gaoya/AAA_test_video/Benchmark/physics_IQ \
  --height 720 \
  --width 1280 \
  --fps 30 \
  --num_frames 240 \
  --seed 42 \
  --model_name wan_22_ti2v_5b


CUDA_VISIBLE_DEVICES=0 python /home/gaoya/Code_Video/physics-IQ-benchmark-main/code/run_physics_iq.py \
  --input_folders /data/gaoya/AAA_test_video/Benchmark/physics_IQ/generated_videos/wan_22_ti2v_5b/ \
  --output_folder /data/gaoya/AAA_test_video/Benchmark/physics_IQ/results/wan_22_ti2v_5b/ \
  --descriptions_file /home/gaoya/Code_Video/physics-IQ-benchmark-main/descriptions/descriptions.csv
 
