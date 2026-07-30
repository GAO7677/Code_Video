#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
CODE_UTILS_ROOT=/home/gaoya/code_my_utils
PYTHON=/mnt/data/gaoya/agent-data/envs/wan-cu128/bin/python
WAN_ROOT=/mnt/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
WEIGHTS_ROOT=/mnt/data/gaoya/agent-data/weights/wan_openvid_lora_step10000
NEGATIVE_PROMPT="模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"

MODEL="${MODEL:?set MODEL}"
SEED="${SEED:?set SEED}"
SUBSET_ID="${SUBSET_ID:?set SUBSET_ID}"
GPU="${GPU:?set GPU}"
STEP_START="${STEP_START:?set STEP_START}"
STEP_END="${STEP_END:?set STEP_END}"
INPUT_LIST="${INPUT_LIST:?set INPUT_LIST}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
MANIFEST="${MANIFEST:?set MANIFEST}"

[[ "${MODEL}" == "openvid_lora_step10000" ]]
[[ "${GPU}" == "6" || "${GPU}" == "7" ]]
(( STEP_START >= 0 && STEP_START < STEP_END && STEP_END <= 40 ))
test -x "${PYTHON}"
test -s "${INPUT_LIST}"
test -s "${MANIFEST}"
test -s "${WEIGHTS_ROOT}/checkpoint.safetensors"

VARIANT="${SUBSET_ID}_steps$(printf '%02d' "${STEP_START}")_$(printf '%02d' "${STEP_END}")"
JOB_ROOT="${OUTPUT_ROOT}/generation/${MODEL}/seed-$(printf '%06d' "${SEED}")/${VARIANT}"
mkdir -p "${JOB_ROOT}"

exec env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${CODE_UTILS_ROOT}:${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}:${SCRIPT_DIR}" \
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" "${SCRIPT_DIR}/infer_openvid_lora_matched_head_ablation.py" \
  --matched-subset-manifest "${MANIFEST}" --matched-subset-id "${SUBSET_ID}" \
  --ablation-step-start "${STEP_START}" --ablation-step-end "${STEP_END}" \
  --weights-root "${WEIGHTS_ROOT}" --input-json-list-path "${INPUT_LIST}" \
  --model-name "openvid_lora_step10000_${VARIANT}" --wan-root "${WAN_ROOT}" \
  --output-root "${JOB_ROOT}/results" --runtime-root "${JOB_ROOT}/runtime" \
  --device cuda --height 512 --width 896 --num-frames 49 \
  --context-frames 8 --conditioning-mode context_aware \
  --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
  --fps 30 --seed "${SEED}" --negative-prompt "${NEGATIVE_PROMPT}" --overwrite
