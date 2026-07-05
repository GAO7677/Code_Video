#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_ROOT="${1:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-007000}"
INPUT_JSON_LIST_PATH="${2:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-7}"

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py \
  --weights-root "${WEIGHTS_ROOT}" \
  --input-json-list-path "${INPUT_JSON_LIST_PATH}" \
  --model-name train_stage1b_diffsynth_native0705_0705_object_context_zero \
  --num-inference-steps 40 \
  --object-context-ablation zero
