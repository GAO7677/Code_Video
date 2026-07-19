

cd /home/gaoya/Code_Video/DiffTrack-main
export PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main/diffusers/src:/home/gaoya/Code_Video/DiffTrack-main
TXT_PROMPTS=/home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/bbb.txt
TXT_TAG=bbb
BASELINE_OUTPUT_DIR=/data/gaoya/agent-data/outputs/difftrack/$TXT_TAG/baseline
PAG_OUTPUT_DIR=/data/gaoya/agent-data/outputs/difftrack/$TXT_TAG/pag

CUDA_VISIBLE_DEVICES=4 /home/gaoya/miniconda3/bin/conda run -n bagel python /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/motion_guidance_5b_local.py \
  --output_dir $PAG_OUTPUT_DIR \
  --model_version 5b \
  --model_path /data/gaoya/ckpt/zai-org-CogVideoX-5b \
  --txt_path $TXT_PROMPTS \
  --pag_layers 15 17 18 \
  --pag_scale 1 \
  --cfg_scale 6 \
  --device cuda:0 \
  --max_prompts 1



cd /home/gaoya/Code_Video/DiffTrack-main
export PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main/diffusers/src:/home/gaoya/Code_Video/DiffTrack-main
TXT_PROMPTS=/home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/bbb.txt
TXT_TAG=bbb
BASELINE_OUTPUT_DIR=/data/gaoya/agent-data/outputs/difftrack/$TXT_TAG/baseline
PAG_OUTPUT_DIR=/data/gaoya/agent-data/outputs/difftrack/$TXT_TAG/pag

CUDA_VISIBLE_DEVICES=5 /home/gaoya/miniconda3/bin/conda run -n bagel python /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/motion_guidance_5b_local.py \
  --output_dir $BASELINE_OUTPUT_DIR \
  --model_version 5b \
  --model_path /data/gaoya/ckpt/zai-org-CogVideoX-5b \
  --txt_path $TXT_PROMPTS \
  --pag_layers 15 17 18 \
  --pag_scale 0 \
  --cfg_scale 6 \
  --device cuda:0 \
  --max_prompts 1

