#!/usr/bin/env bash
set -euo pipefail

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan-cu128

cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt

# True 2-card DDP: one process per GPU (rank0->gpu6, rank1->gpu7).
export CUDA_VISIBLE_DEVICES=2,3,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/object_token_teacher_student/config_stage1b_oracle_cross_attn_template.yaml

accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  -m code_vjepa_vggt.object_token_teacher_student.train_stage1b_oracle_cross_attn \
  --config "${CONFIG}"
