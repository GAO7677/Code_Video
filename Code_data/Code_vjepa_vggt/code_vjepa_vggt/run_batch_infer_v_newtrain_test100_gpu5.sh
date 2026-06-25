#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main
export CUDA_VISIBLE_DEVICES=5
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/batch_infer_v_newtrain_from_jsonl.py \
  --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_100.txt \
  --checkpoint-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints \
  --steps step-001200 step-001600 \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml \
  --infer-script /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_v_newtrain_context_video_wan.py \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/pybullet0624_diffsynth_object_v_newtrain_gpu67 \
  --python-exe /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  --num-frames 24 \
  --context-frames 8 \
  --sampling-mode prefix \
  --sampling-steps 40 \
  --fps 30 \
  --seed 42 \
  --cfg-scale 5.0
