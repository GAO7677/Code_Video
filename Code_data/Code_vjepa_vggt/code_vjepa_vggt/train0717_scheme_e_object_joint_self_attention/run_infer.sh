#!/usr/bin/env bash
set -euo pipefail

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
PROJECT=${BASE}/code_vjepa_vggt/train0717_scheme_e_object_joint_self_attention
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

: "${WEIGHTS_ROOT:?Set WEIGHTS_ROOT to a Scheme-E step-* directory}"
: "${INPUT_JSON_LIST:?Set INPUT_JSON_LIST to a txt file containing input JSON paths}"
GPU_PAIR="${GPU_PAIR:-7}"
INFERENCE_DEVICES="${INFERENCE_DEVICES:-cuda:0,cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/scheme_e_joint_self_attn}"
MODEL_NAME="${MODEL_NAME:-scheme_e_joint_self_attn_$(basename "${WEIGHTS_ROOT}")}"

env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${BASE}:${DIFFSYNTH_ROOT}" \
  DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_PAIR}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SCHEME_D_TUBE_NUM_TOKENS=4 \
  SCHEME_D_TUBE_HIDDEN_DIM=256 \
  SCHEME_D_TUBE_MOTION_TOKENS=4 \
  SCHEME_D_TUBE_MOTION_FOURIER_BANDS=4 \
  SCHEME_D_TUBE_OBJECT_ATTN_DIM=256 \
  SCHEME_D_TUBE_OBJECT_ATTN_HEADS=8 \
  SCHEME_D_OBJECT_BLOCK_IDS=8,14,20 \
  SCHEME_D_ENTITY_BINDING_BOTTLENECK_DIM=256 \
  SCHEME_D_ENTITY_BINDING_GATE_INIT=0.5 \
  SCHEME_D_ENTITY_BINDING_RESIDUAL_MAX_RATIO=0.5 \
  SCHEME_D_OBJECT_INPUT_ABLATION="${SCHEME_E_OBJECT_INPUT_ABLATION:-none}" \
  "${PYTHON}" "${PROJECT}/infer.py" \
    --weights-root "${WEIGHTS_ROOT}" \
    --input-json-list-path "${INPUT_JSON_LIST}" \
    --model-name "${MODEL_NAME}" \
    --output-root "${OUTPUT_ROOT}" --step-output-dir-name results \
    --inference-devices "${INFERENCE_DEVICES}" \
    --height 512 --width 896 --context-frames 8 --num-frames 49 \
    --num-inference-steps "${NUM_INFERENCE_STEPS:-40}" \
    --cfg-scale "${CFG_SCALE:-5.0}" --seed "${SEED:-42}" --fps 30 \
    --object-num-queries 8 --aux-max-objects 4 --object-pooler-latent-dim 48 \
    --cond-proj-dim 256 --compact-object-context-slots \
    --object-adapter-mlp-residual-max-ratio 3.0 \
    --object-context-ablation "${OBJECT_CONTEXT_ABLATION:-none}" \
    --object-context-scale-factor 1.0 \
    --object-branch-residual-scale "${OBJECT_BRANCH_RESIDUAL_SCALE:-1.0}" \
    --object-branch-ratio-guard-max-ratio 0.30 \
    --object-branch-ratio-guard-max-block-id -1 \
    --grounding-text-prompt "" --grounding-enable-caption-terms \
    --grounding-caption-prompt-mode physical_noun_phrases \
    --grounding-caption-max-phrases 4 --grounding-caption-min-score 4.0 \
    --grounding-gdino-box-threshold 0.20 --grounding-gdino-text-threshold 0.15 \
    --grounding-prompt-frame-mode first --sam2-segment-len 8 --force

echo "inference output: ${OUTPUT_ROOT}"
