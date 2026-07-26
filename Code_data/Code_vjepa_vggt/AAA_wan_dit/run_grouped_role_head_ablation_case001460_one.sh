#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash run_grouped_role_head_ablation_case001460_one.sh MODEL CATEGORY GPU_ID
# MODEL: wan_lora | xssc | physrvg
# CATEGORY: S | T | P | C | G

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 MODEL CATEGORY GPU_ID" >&2
  exit 2
fi

MODEL="$1"
CATEGORY="${2^^}"
GPU_ID="$3"
case "${MODEL}" in wan_lora|xssc|physrvg) ;; *) exit 2 ;; esac
case "${CATEGORY}" in S|T|P|C|G) ;; *) exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PHYSRVG_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main
WAN_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PHYSRVG_PYTHON=/data/gaoya/miniconda3/envs/vjepa2/bin/python

INPUT_LIST="${SCRIPT_DIR}/ball_query_case001460.txt"
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/agent-data/outputs/wan_dit_grouped_role_head_ablation/case001460}"
TAG="self_attn_grouped_head_zero_category_${CATEGORY,,}"
OUTPUT_ROOT="${OUTPUT_BASE}/${MODEL}/${TAG}"

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

mkdir -p "${OUTPUT_ROOT}"
test -s "${INPUT_LIST}"
{
  echo "model=${MODEL}"
  echo "category=${CATEGORY}"
  echo "gpu=${GPU_ID}"
  echo "input_list=${INPUT_LIST}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "height=512"
  echo "width=896"
  echo "num_frames=49"
  echo "context_frames=8"
  echo "num_inference_steps=40"
  echo "seed=42"
  echo "targets_source=${SCRIPT_DIR}/grouped_head_targets.py"
} > "${OUTPUT_ROOT}/ablation_config.txt"

COMMON_ENV=(
  PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

if [[ "${MODEL}" == "wan_lora" ]]; then
  exec env "${COMMON_ENV[@]}" \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/infer_wan_lora_grouped_head_ablation.py" \
    --grouped-head-category "${CATEGORY}" \
    --weights-root "${WAN_LORA_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name "wan_lora_${TAG}" \
    --wan-root "${WAN_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --runtime-root "${OUTPUT_ROOT}/_runtime" \
    --device cuda --height 512 --width 896 --num-frames 49 \
    --context-frames 8 --conditioning-mode context_aware \
    --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed 42 --negative-prompt "${NEGATIVE_PROMPT}"
fi

if [[ "${MODEL}" == "xssc" ]]; then
  exec env "${COMMON_ENV[@]}" \
    XSSC_ROOT="${XSSC_ROOT}" XSSC_CONFIG="${XSSC_CONFIG}" \
    XSSC_CHECKPOINT="${XSSC_CHECKPOINT}" XSSC_PREPROCESS_MODE=center_crop \
    XSSC_SLOT_TEMPORAL_MODE=full \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/infer_xssc_grouped_head_ablation.py" \
    --grouped-head-category "${CATEGORY}" \
    --weights-root "${XSSC_WEIGHTS_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name "xssc_${TAG}" \
    --output-root "${OUTPUT_ROOT}" --step-output-dir-name results \
    --wan-root "${WAN_ROOT}" \
    --lora-checkpoint "${WAN_LORA_ROOT}/checkpoint.safetensors" \
    --device cuda:0 --aux-device cuda:0 --inference-devices cuda:0,cuda:0 \
    --height 512 --width 896 --num-frames 49 --context-frames 8 \
    --sampling-mode prefix --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed 42 --negative-prompt "${NEGATIVE_PROMPT}"
fi

exec env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH="${PHYSRVG_ROOT}:${SCRIPT_DIR}" \
  PYTHONNOUSERSITE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PHYSRVG_PYTHON}" "${SCRIPT_DIR}/infer_physrvg_dit_ablation.py" \
  --physrvg-grouped-head-category "${CATEGORY}" \
  --expected-context-frames 8 --physrvg-root "${PHYSRVG_ROOT}" \
  --input-json-list-paths "${INPUT_LIST}" --output-root "${OUTPUT_ROOT}" \
  --model-id "${MODEL_ID}" --dit-checkpoint "${DIT_CHECKPOINT}" \
  --lora-checkpoint "${LORA_CHECKPOINT}" --device cuda:0 \
  --height 512 --width 896 --num-frames 49 --fps 30 \
  --num-inference-steps 40 --guidance-scale 5.0 --seed 42 --force
