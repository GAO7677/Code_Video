#!/usr/bin/env bash
set -euo pipefail

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SCRIPT=${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/visualize_caption_grounding_track_v2v.py
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/caption_grounding_track_physiq4_20260714}

env PYTHONNOUSERSITE=1 \
  PYTHONPATH=${PROJ} \
  CUDA_VISIBLE_DEVICES=${GPU:-6} \
  ${PYTHON} ${SCRIPT} \
  --output-root ${OUTPUT_ROOT} \
  --device cuda:0 \
  --num-context-frames 8 --height 512 --width 896 --fps 12 \
  --max-objects 4 --points-per-object 8 \
  --caption-max-phrases 4 --caption-min-score 4.0 \
  --gdino-box-threshold 0.20 --gdino-text-threshold 0.15 \
  --input-json /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end.json \
  --input-json /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json \
  --input-json /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json \
  --input-json /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px.json
