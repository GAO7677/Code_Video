
cd /home/gaoya/Code_Video/phaselock-main/code
CUDA_VISIBLE_DEVICES=7 /data/gaoya/miniconda3/envs/wan/bin/
python /home/gaoya/Code_Video/phaselock-main/code/scripts/wan_ti2v_phaselock.py --ckpt_dir /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B 
--size 1280*704 
--prompt "A ball hits a block of wood." 
--output /
home/gaoya/Code_Video/phaselock-main/outputs/wan_ti2v_phaselock_ball_hits_wood_gpu7.mp4 
--few_steps 2 
--full_steps 50 
--frame_num 121 
--seed 42 
--offload_model 
--t5_cpu 
--convert_model_dtype
--device_id 0


cd /home/gaoya/Code_Video/WAN_2p2/Wan2.2-main/
CUDA_VISIBLE_DEVICES=0 python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt "A ball hits a block of wood."

