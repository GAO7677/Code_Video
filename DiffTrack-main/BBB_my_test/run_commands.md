
```bash
cd /home/gaoya/Code_Video/DiffTrack-main
export PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main/diffusers/src:/home/gaoya/Code_Video/DiffTrack-main

CUDA_VISIBLE_DEVICES=4 /home/gaoya/miniconda3/bin/conda run -n bagel python /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/motion_guidance_5b_local.py \
  --output_dir /data/gaoya/agent-data/outputs/difftrack_diag_case0_orig_pag151718 \
  --model_version 5b \
  --model_path /data/gaoya/ckpt/zai-org-CogVideoX-5b \
  --txt_path /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/prompt_rigid_case0_original.txt \
  --pag_layers 15 17 18 \
  --pag_scale 1 \
  --cfg_scale 6 \
  --device cuda:0 \
  --max_prompts 1

CUDA_VISIBLE_DEVICES=5 /home/gaoya/miniconda3/bin/conda run -n bagel python /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/motion_guidance_5b_local.py \
  --output_dir /data/gaoya/agent-data/outputs/difftrack_diag_case0_simple_baseline \
  --model_version 5b \
  --model_path /data/gaoya/ckpt/zai-org-CogVideoX-5b \
  --txt_path /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/prompt_rigid_case0_simple.txt \
  --pag_layers 15 17 18 \
  --pag_scale 0 \
  --cfg_scale 6 \
  --device cuda:0 \
  --max_prompts 1

CUDA_VISIBLE_DEVICES=6 /home/gaoya/miniconda3/bin/conda run -n bagel python /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/motion_guidance_5b_local.py \
  --output_dir /data/gaoya/agent-data/outputs/difftrack_diag_case0_simple_pag151718 \
  --model_version 5b \
  --model_path /data/gaoya/ckpt/zai-org-CogVideoX-5b \
  --txt_path /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/prompt_rigid_case0_simple.txt \
  --pag_layers 15 17 18 \
  --pag_scale 1 \
  --cfg_scale 6 \
  --device cuda:0 \
  --max_prompts 1
```
