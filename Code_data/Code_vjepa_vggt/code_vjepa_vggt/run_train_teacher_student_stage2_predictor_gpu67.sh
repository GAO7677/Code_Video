#!/usr/bin/env bash
set -euo pipefail

cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt

export CUDA_VISIBLE_DEVICES=6,7

CONFIG=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/object_token_teacher_student/config_stage2_predictor_template.yaml

accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  -m code_vjepa_vggt.object_token_teacher_student.train_stage2_predictor \
  --config "${CONFIG}"
