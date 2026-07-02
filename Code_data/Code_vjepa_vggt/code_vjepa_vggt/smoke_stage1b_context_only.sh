#!/usr/bin/env bash
# Smoke test: train 3 steps -> resume from step 1 -> run inference.
# Uses GPU 5 only (single-card, fast startup).
set -euo pipefail

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan-cu128

cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/object_token_teacher_student/config_stage1b_context_only_smoke_resume.yaml
INIT_FROM=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt
SMOKE_CKPT_DIR=/data/gaoya/agent-data/checkpoints/pybullet0629_teacher_student/stage1b_context_only_smoke
TEST_JSON_LIST=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt

echo "============================================"
echo "STEP 1: Train from scratch (steps 1-3)"
echo "============================================"
rm -rf "${SMOKE_CKPT_DIR}"
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  -m code_vjepa_vggt.object_token_teacher_student.train_stage1b_context_only \
  --config "${CONFIG}" \
  --init-from "${INIT_FROM}"

echo ""
echo "Checkpoints after first run:"
ls -lh "${SMOKE_CKPT_DIR}"/step_*.pt 2>/dev/null || echo "  (none found)"

RESUME_CKPT=$(ls "${SMOKE_CKPT_DIR}"/step_*.pt 2>/dev/null | sort | head -1)
if [[ -z "${RESUME_CKPT}" ]]; then
  echo "ERROR: no checkpoint found after first run, aborting."
  exit 1
fi

echo ""
echo "============================================"
echo "STEP 2: Resume from ${RESUME_CKPT}"
echo "============================================"
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  -m code_vjepa_vggt.object_token_teacher_student.train_stage1b_context_only \
  --config "${CONFIG}" \
  --init-from "${INIT_FROM}" \
  --resume-checkpoint "${RESUME_CKPT}"

echo ""
echo "Checkpoints after resume run:"
ls -lh "${SMOKE_CKPT_DIR}"/step_*.pt 2>/dev/null || echo "  (none found)"

INFER_CKPT=$(ls "${SMOKE_CKPT_DIR}"/step_*.pt 2>/dev/null | sort | tail -1)
if [[ -z "${INFER_CKPT}" ]]; then
  echo "ERROR: no checkpoint found for inference, aborting."
  exit 1
fi

echo ""
echo "============================================"
echo "STEP 3: Inference with ${INFER_CKPT}"
echo "============================================"
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
python3 /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_stage1b_context_only_v2v.py \
  --checkpoint "${INFER_CKPT}" \
  --init-from "${INIT_FROM}" \
  --input-json-list-path "${TEST_JSON_LIST}" \
  --model-name smoke_context_only \
  --sampling-steps 10 \
  --limit 1 \
  --save-raw \
  --force

echo ""
echo "============================================"
echo "SMOKE TEST PASSED"
echo "============================================"
