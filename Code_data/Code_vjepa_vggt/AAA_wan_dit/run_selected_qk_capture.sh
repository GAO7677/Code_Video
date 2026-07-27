#!/usr/bin/env bash
set -euo pipefail

# MODEL=wan_lora GPU=0 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_selected_qk_capture.sh
# MODEL=xssc GPU=2 INPUT_LIST=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/fulltoken_moving_xssc_replay_prefix12.txt bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_selected_qk_capture.sh
# MODEL=physrvg GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_selected_qk_capture.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PHYSRVG_ROOT="${PROJECT_ROOT}/code_phys_papers_compare/PhysRVG-main"
WAN_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PHYSRVG_PYTHON=/data/gaoya/miniconda3/envs/vjepa2/bin/python

MODEL="${MODEL:?set MODEL to wan_lora, xssc, or physrvg}"
GPU="${GPU:?set one physical GPU id}"
SEED="${SEED:-851}"
ROOT="${ROOT:-/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/alltoken_qk}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/capture/${MODEL}}"
SELECTION="${SELECTION:-${ROOT}/selection.json}"
INPUT_LIST="${INPUT_LIST:-${SCRIPT_DIR}/fulltoken_moving_pilot_cases.txt}"
STEPS="${STEPS:-5,15,25,35}"
OUTPUT_BINS="${OUTPUT_BINS:-512}"
QUERY_CHUNK="${QUERY_CHUNK:-64}"

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
WAN_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500
XSSC_WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots/checkpoints/step-001500
XSSC_ROOT=/home/gaoya/Code_Video/xSSC-main
XSSC_CONFIG="${XSSC_ROOT}/config-randsfq/rsfq2_r-ytvis.py"
XSSC_CHECKPOINT=/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth
MODEL_ID=/data/gaoya/ckpt/HappyP4nda-PhysRVG/Wan2.2-TI2V-5B-Diffusers
DIT_CHECKPOINT=/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors
LORA_CHECKPOINT=/data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint
NEGATIVE_PROMPT="模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"

mkdir -p "${OUTPUT_ROOT}" "${ROOT}/logs"
test -s "${SELECTION}"
test -s "${INPUT_LIST}"
QK_ARGS=(
  --qk-output-root "${OUTPUT_ROOT}"
  --qk-selection "${SELECTION}"
  --qk-steps "${STEPS}"
  --qk-output-bins "${OUTPUT_BINS}"
  --qk-query-chunk "${QUERY_CHUNK}"
)

if [[ "${MODEL}" == "wan_lora" ]]; then
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/capture_selected_qk_wan_lora.py" \
    "${QK_ARGS[@]}" --weights-root "${WAN_LORA_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name wan_lora_selected_qk --wan-root "${WAN_ROOT}" \
    --output-root "${OUTPUT_ROOT}/generated/wan_lora" \
    --runtime-root "${OUTPUT_ROOT}/generated/wan_lora/_runtime" \
    --device cuda --height 512 --width 896 --num-frames 49 \
    --context-frames 8 --conditioning-mode context_aware \
    --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed "${SEED}" --negative-prompt "${NEGATIVE_PROMPT}" --overwrite
elif [[ "${MODEL}" == "xssc" ]]; then
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    XSSC_ROOT="${XSSC_ROOT}" XSSC_CONFIG="${XSSC_CONFIG}" \
    XSSC_CHECKPOINT="${XSSC_CHECKPOINT}" \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/capture_selected_qk_xssc.py" \
    "${QK_ARGS[@]}" --weights-root "${XSSC_WEIGHTS_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name xssc_selected_qk \
    --output-root "${OUTPUT_ROOT}/generated/xssc" \
    --step-output-dir-name results --wan-root "${WAN_ROOT}" \
    --lora-checkpoint "${WAN_LORA_ROOT}/checkpoint.safetensors" \
    --device cuda:0 --aux-device cuda:0 --inference-devices cuda:0,cuda:0 \
    --height 512 --width 896 --num-frames 49 --context-frames 8 \
    --sampling-mode prefix --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed "${SEED}" --negative-prompt "${NEGATIVE_PROMPT}" --overwrite
elif [[ "${MODEL}" == "physrvg" ]]; then
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONPATH="${PHYSRVG_ROOT}:${SCRIPT_DIR}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PHYSRVG_PYTHON}" "${SCRIPT_DIR}/capture_selected_qk_physrvg.py" \
    "${QK_ARGS[@]}" --physrvg-root "${PHYSRVG_ROOT}" \
    --input-json-list-paths "${INPUT_LIST}" \
    --output-root "${OUTPUT_ROOT}/generated/physrvg" \
    --model-id "${MODEL_ID}" --dit-checkpoint "${DIT_CHECKPOINT}" \
    --lora-checkpoint "${LORA_CHECKPOINT}" --device cuda:0 \
    --height 512 --width 896 --num-frames 49 --fps 30 \
    --num-inference-steps 40 --guidance-scale 5.0 --seed "${SEED}" --force
else
  echo "unsupported MODEL=${MODEL}" >&2
  exit 2
fi
