#!/usr/bin/env bash
set -euo pipefail

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan-cu128

cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
export CUDA_VISIBLE_DEVICES=${GPU:-3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/config_stage1a_full_token_smoke_wan21_13b.yaml
INIT_FROM=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/smoke/raw_phys_state_lora_continue/checkpoints/step-000002/checkpoint.safetensors

python -m code_vjepa_vggt.object_token_teacher_student.train_stage1a_full_token \
  --config "${CONFIG}" \
  --init-from "${INIT_FROM}" \
  "$@"
