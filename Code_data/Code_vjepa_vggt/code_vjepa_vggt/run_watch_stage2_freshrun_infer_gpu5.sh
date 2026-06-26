#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/watch_checkpoint_infer.py \
  --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun/checkpoints \
  --infer-script /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_v_newtrain_context_video_wan.py \
  --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 \
  --prompt "industrial rigid body simulation sphere" \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_test/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun \
  --state-file /data/gaoya/AAA_test_video/0623/train/train0624/infer_test/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun/watch_state.json \
  --gpu 5 \
  --num-frames 24 \
  --sampling-mode prefix \
  --sampling-steps 40 \
  --poll-seconds 30
