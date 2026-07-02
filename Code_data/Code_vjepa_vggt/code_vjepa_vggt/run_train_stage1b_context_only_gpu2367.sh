#!/usr/bin/env bash
set -euo pipefail

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan-cu128

cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
export CUDA_VISIBLE_DEVICES=2,3,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/object_token_teacher_student/config_stage1b_context_only_template.yaml
STAGE1A_CKPT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt
CKPT_DIR=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1b_context_only

# Auto-detect latest checkpoint for resume
RESUME_ARG=""
if [ -d "${CKPT_DIR}" ]; then
    LATEST=$(ls "${CKPT_DIR}"/step_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "${LATEST}" ]; then
        echo "[resume] found checkpoint: ${LATEST}"
        RESUME_ARG="--resume-checkpoint ${LATEST}"
    else
        echo "[fresh] no checkpoint found, starting from scratch"
    fi
fi

accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m code_vjepa_vggt.object_token_teacher_student.train_stage1b_context_only \
    --config "${CONFIG}" \
    --init-from "${STAGE1A_CKPT}" \
    ${RESUME_ARG} \
    "$@"
