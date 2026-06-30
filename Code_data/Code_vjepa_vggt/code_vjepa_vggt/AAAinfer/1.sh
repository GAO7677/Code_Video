
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_vnewtrain_0613pybullet_stage2_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun/checkpoints/step-007500 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun \
  

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_vnewtrain_0613pybullet_stage2_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name pybullet0624_diffsynth_object_v_newtrain_gpu67 \
  
