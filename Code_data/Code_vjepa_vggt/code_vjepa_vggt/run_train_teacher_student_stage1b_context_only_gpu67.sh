#!/usr/bin/env bash
set -euo pipefail

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan-cu128

cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt

export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CODEX_DEBUG_TRAINER_INIT=1
export CODEX_DEBUG_RUNNER_INIT=1

CONFIG=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/object_token_teacher_student/config_stage1b_context_only_smoke.yaml
INIT_FROM=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt

accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  -m code_vjepa_vggt.object_token_teacher_student.train_stage1b_context_only \
  --config "${CONFIG}" \
  --init-from "${INIT_FROM}" \
  "$@"
