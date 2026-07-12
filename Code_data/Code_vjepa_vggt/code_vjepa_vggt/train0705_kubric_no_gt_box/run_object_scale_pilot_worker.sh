#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU SCALE TAG" >&2
  exit 2
fi

PHYSICAL_GPU="$1"
SCALE="$2"
TAG="$3"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WRAPPER=${PROJECT_ROOT}/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v_queryscheme.py
WEIGHTS=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708_stability_v3_from_scratch_20260711T144000Z/checkpoints/step-003500
CASE_LIST=/data/gaoya/agent-data/outputs/physiq_full_scheme_review_20260712/selected_input_jsons.txt
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/physiq_full_scheme_review_20260712/scale_sweep
NEGATIVE_PROMPT='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" "${WRAPPER}" \
  --query-scheme temporal_sam2 \
  --weights-root "${WEIGHTS}" \
  --input-json-list-path "${CASE_LIST}" \
  --model-name "stability_v3_step3500_temporal_scale_${TAG}" \
  --output-root "${OUTPUT_ROOT}" \
  --step-output-dir-name __METHOD_NAME__ \
  --method-suffix "temporal_scale_${TAG}" \
  --inference-devices cuda:0,cuda:0 \
  --num-frames 49 --context-frames 8 --sampling-mode prefix \
  --num-inference-steps 40 --cfg-scale 5.0 --seed 42 \
  --height 512 --width 896 --input-cover-crop-height 512 --input-cover-crop-width 896 \
  --fps 30 --negative-prompt "${NEGATIVE_PROMPT}" \
  --compact-object-context-slots \
  --object-context-scale-factor "${SCALE}" \
  --object-adapter-mlp-residual-max-ratio 3.0 \
  --object-branch-ratio-guard-max-ratio 0.30 \
  --object-branch-ratio-guard-max-block-id -1
