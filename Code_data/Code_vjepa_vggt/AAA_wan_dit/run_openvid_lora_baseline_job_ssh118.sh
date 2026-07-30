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

GPU="${GPU:?set GPU}"
SEED="${SEED:?set SEED}"
INPUT_LIST="${INPUT_LIST:?set INPUT_LIST}"
JOB_ROOT="${JOB_ROOT:?set JOB_ROOT}"

[[ "${GPU}" == "6" || "${GPU}" == "7" ]]
test -x "${PYTHON}"
test -s "${INPUT_LIST}"
test -s "${WEIGHTS_ROOT}/checkpoint.safetensors"
mkdir -p "${JOB_ROOT}"

exec env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${CODE_UTILS_ROOT}:${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}:${SCRIPT_DIR}" \
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" "${SCRIPT_DIR}/infer_openvid_lora_dit_ablation.py" \
  --dit-ablation-mode baseline \
  --weights-root "${WEIGHTS_ROOT}" --input-json-list-path "${INPUT_LIST}" \
  --model-name openvid_lora_step10000_baseline --wan-root "${WAN_ROOT}" \
  --output-root "${JOB_ROOT}/results" --runtime-root "${JOB_ROOT}/runtime" \
  --device cuda --height 512 --width 896 --num-frames 49 \
  --context-frames 8 --conditioning-mode context_aware \
  --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
  --fps 30 --seed "${SEED}" --negative-prompt "${NEGATIVE_PROMPT}"
