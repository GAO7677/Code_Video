#!/usr/bin/env bash
set -euo pipefail

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
SCRIPT=${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_entity_id_binding_visualized_v2v.py
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/entity_id_binding_physiq3_current_20260714}
INPUT_LIST=${OUTPUT_ROOT}/input_jsons.txt
CHECKPOINT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_init3500_save500_keepall_20260713T090024Z/checkpoints/step-003500

env PYTHONNOUSERSITE=1 \
  PYTHONPATH=${PROJ}:${DIFFSYNTH} \
  CUDA_VISIBLE_DEVICES=${GPU:-2} \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ${PYTHON} ${SCRIPT} \
  --weights-root ${CHECKPOINT} \
  --input-json-list-path ${INPUT_LIST} \
  --model-name entity_id_binding_current_mixdataset_step3500 \
  --output-root ${OUTPUT_ROOT} \
  --step-output-dir-name results \
  --device cuda:0 --aux-device cuda:0 \
  --height 512 --width 896 --input-cover-crop-height 512 --input-cover-crop-width 896 \
  --num-frames 49 --context-frames 8 --sampling-mode prefix \
  --num-inference-steps 40 --cfg-scale 5.0 --seed 42 --fps 30 \
  --compact-object-context-slots \
  --object-adapter-mlp-residual-max-ratio 3.0 \
  --object-branch-residual-scale 1.0 \
  --object-branch-ratio-guard-max-ratio 0.20 \
  --object-branch-ratio-guard-max-block-id -1 \
  --grounding-proposal-source gdino_only \
  --grounding-text-prompt "box . cube . block . cylinder . capsule . sphere . ball . person . car . vehicle . container ." \
  --grounding-disable-caption-terms \
  --grounding-gdino-box-threshold 0.20 --grounding-gdino-text-threshold 0.15 \
  --grounding-prompt-frame-mode first --grounding-track-dedupe-iou-threshold 0.75 \
  --grounding-container-suppress-ratio-threshold 0.95 \
  --grounding-container-suppress-min-contained 2 \
  --grounding-container-suppress-min-area-ratio 1.5 \
  --grounding-container-suppress-small-iou-threshold 0.7 \
  --sam2-segment-len 8 --force
